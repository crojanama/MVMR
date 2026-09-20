# script6_ses_lambda_function.py for reports-inbound
#
# SCRIPT 6 — SES-triggered ingest.  Resend is gone.
#
# WHAT CHANGED FROM SCRIPT 5:
#   * Trigger is now an Amazon SES receipt rule, not a Resend webhook.  The rule
#     has two ordered actions:
#         1. Deliver to S3   -> writes the raw MIME to  reports-inbound/raw/{messageId}
#         2. Invoke Lambda   -> this function (Event / async invocation)
#     Because the S3 action runs first, the raw email is already in the bucket by
#     the time we run; we read and parse it ourselves.
#   * No Resend API, no RESEND_API_KEY, no attachment download URLs.  Attachments
#     and inline chart images come straight out of the MIME with the stdlib
#     `email` package.
#   * ONE bucket for everything (reports-inbound).  Routing that used to pick a
#     bucket now picks a PREFIX inside the same bucket.
#
# 2026-08-27 — THIRD REPORT TYPE: River Metals Recycling (RMR) NF price sheet.
#   A weekly pricing PDF arrives on its own email.  It is stored, then two
#   DISCOUNTED copies are generated (20% and 18% off by default) and stored
#   beside it, ready to be emailed on to an associate (send step is separate).
#
#     reports-inbound/RMR/YYYY/MM/DD/River_Metals_NF_Pricing_8.26.26.pdf     (original)
#     reports-inbound/RMR/YYYY/MM/DD/River_Metals_NF_Pricing_8.26.26_20.pdf
#     reports-inbound/RMR/YYYY/MM/DD/River_Metals_NF_Pricing_8.26.26_18.pdf
#
#   The discounted sheets are the ORIGINAL PDF edited in place — every price is
#   redacted and rewritten at the same coordinates — so the layout, logo, table
#   shading and item codes are identical to the source.  Two things are
#   deliberately NOT discounted:
#     * the Comex / LME figures in the header: those are market index quotes,
#       not our prices (gated out by RMR_TABLE_TOP_Y).
#     * "call for price" rows: no number to discount, left verbatim.
#   Each generated sheet carries a single grey footnote under the table naming
#   the discount; there is no banner at the top of the page, so the sheet is
#   visually identical to the published original apart from the numbers.
#
# 2026-09-10 - RMR naming + labelling cleanup.
#   * Filenames are normalised: River Metals mails the sheet with a random
#     per-send token ("..._8.26.26_f_mtw1d3.pdf").  normalize_rmr_filename()
#     trims everything after the M.D.YY stamp, and the discounted copies are
#     suffixed "_20" / "_18" instead of "_LESS20" / "_LESS18".
#   * The red "LESS 20%" banner drawn on each discounted sheet is gone.
#   * The "Market quotes in the header ..." paragraph is out of the outbound
#     email body.
#   Sheets stored before this date keep their old "_LESS20"/"_LESS18" names;
#   run_rmr_regenerate on the original writes the new names beside them.
#
# STORAGE LAYOUT
#     parsed reports   -> {MAIN_PREFIX}/YYYY/MM/DD/...   (bucket root by default)
#     RMR price sheets -> RMR/YYYY/MM/DD/...             (RMR_PREFIX)
#     raw email        -> raw/{messageId}                (RAW_PREFIX)
#     status JSON      -> _job-status/YYYY/MM/DD/{messageId}.json
#     debug text/kpi   -> _debug-text/... , _debug-kpi/...
#
# REQUIRES on the execution role:
#   s3:GetObject  on  reports-inbound/*          (read raw email + stored PDFs)
#   s3:PutObject  on  reports-inbound/*          (write reports/status/debug)
#   s3:ListBucket on  reports-inbound            (cleanup / catch-up modes)
#   textract:DetectDocumentText
#   PyMuPDF (fitz) must be available via a layer, same as Script 5.
#
# MANUAL TEST EVENTS (run from the Lambda console):
#   {"run_reprocess_key": "raw/<messageId>"}        -> reparse one stored email
#   {"run_ocr_test_key":  "raw/<messageId>"}        -> OCR its images, write nothing
#   {"run_rmr_test_key":  "raw/<messageId>"}        -> report what the RMR path WOULD
#                                                      do (prices found + samples),
#                                                      writes nothing
#   {"run_rmr_regenerate": "RMR/2026/08/26/<file>.pdf"}
#                                                   -> rebuild the discounted sheets
#                                                      from an already-stored original
#   {"run_kpi_cleanup": true, "dry_run": true}      -> list blank KPI CSVs
#   {"run_catchup": {"since": "YYYY-MM-DD", "until": "YYYY-MM-DD"}}
import csv
import email
import email.policy
import email.utils
import hashlib
import io
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

import boto3
import fitz  # PyMuPDF

s3 = boto3.client("s3")
textract = boto3.client("textract")
# SES v2: 40 MB message limit (v1 SendRawEmail caps at 10 MB).  The three price
# sheets total ~2.5 MB after base64, so either would do — v2 is future-proof.
sesv2 = boto3.client("sesv2")

S3_BUCKET = os.environ.get("S3_BUCKET") or os.environ.get("HS_S3_BUCKET") or "reports-inbound"
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw/")

MAIN_PREFIX = os.environ.get("MAIN_PREFIX", "")
RAW_PDF_PREFIX = os.environ.get("RAW_PDF_PREFIX", "")
STATUS_PREFIX = os.environ.get("STATUS_PREFIX", "_job-status")
DEBUG_TEXT_PREFIX = os.environ.get("DEBUG_TEXT_PREFIX", "_debug-text")
DEBUG_KPI_PREFIX = os.environ.get("DEBUG_KPI_PREFIX", "_debug-kpi")

# ---- River Metals (RMR) price-sheet settings -------------------------
RMR_PREFIX = os.environ.get("RMR_PREFIX", "RMR")
# Comma-separated discounts to generate, e.g. "20,18".
RMR_DISCOUNTS = [
    float(p) for p in (os.environ.get("RMR_DISCOUNTS", "20,18") or "").split(",") if p.strip()
]
# Substring matched against the SENDER to identify the price-sheet email.
# Set this to the real sending domain once known; the subject/filename checks
# below already catch the current sheets on their own.
RMR_SENDER_HINT = os.environ.get("RMR_SENDER_HINT", "").strip().lower()
# Anything ABOVE this y-coordinate on the page is header (the Comex / LME
# market quotes) and must never be discounted.  The price table starts at
# y=136 on the observed sheet; the header quotes sit at y=76-99.
RMR_TABLE_TOP_Y = float(os.environ.get("RMR_TABLE_TOP_Y", "120"))

# ---- RMR outbound email (step 2) -------------------------------------
# After the sheets are generated they are emailed on automatically.  Sending is
# gated on RMR_AUTO_SEND *and* a non-empty RMR_MAIL_TO, so an unconfigured
# function stores the PDFs and simply skips the send rather than erroring.
RMR_AUTO_SEND = os.environ.get("RMR_AUTO_SEND", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RMR_MAIL_FROM = os.environ.get("RMR_MAIL_FROM", "noreply@miamivalleyrecycling.com").strip()
# Comma-separated recipient list.
RMR_MAIL_TO = [a.strip() for a in os.environ.get("RMR_MAIL_TO", "").split(",") if a.strip()]
RMR_MAIL_CC = [a.strip() for a in os.environ.get("RMR_MAIL_CC", "").split(",") if a.strip()]
# noreply@ is not a real mailbox, so point replies somewhere a human reads.
RMR_MAIL_REPLY_TO = [
    a.strip() for a in os.environ.get("RMR_MAIL_REPLY_TO", "").split(",") if a.strip()
]
# Attach the untouched original alongside the two discounted sheets.
RMR_ATTACH_ORIGINAL = os.environ.get("RMR_ATTACH_ORIGINAL", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# SES v2 hard limit is 40 MB after encoding; stop well short and report clearly.
RMR_MAX_MESSAGE_BYTES = int(os.environ.get("RMR_MAX_MESSAGE_BYTES", str(35 * 1024 * 1024)))

HUBSPOT_TOKEN = os.environ.get("HUBSPOT_TOKEN", "")
KPI_SOURCE = os.environ.get("KPI_SOURCE", "ocr").lower()
HUBSPOT_API_BASE = "https://api.hubapi.com"
USER_AGENT = "reports-inbound-ses-ingest/1.0"

MIN_CONFIDENCE = float(os.environ.get("KPI_MIN_CONFIDENCE", "90"))

KPI_METRICS = [
    (
        "new_calls_last_week",
        "New Calls Created Last Week",
        r"New\s+Calls\s+Created\s+Last\s+Week",
        "calls",
        "hs_createdate",
    ),
    (
        "new_contacts_last_week",
        "New Contacts Added Last Week",
        r"New\s+Contacts\s+Added\s+Last\s+Week",
        "contacts",
        "createdate",
    ),
    (
        "new_meetings_last_week",
        "New Meetings Created Last Week",
        r"New\s+Meetings\s+Created\s+Last\s+Week",
        "meetings",
        "hs_createdate",
    ),
]


# =====================================================================
# Entry point
# =====================================================================
def lambda_handler(event, context):
    try:
        # ---- manual test / operational modes ----
        if event.get("run_reprocess_key"):
            return reprocess_raw_key(event["run_reprocess_key"])
        if event.get("run_ocr_test_key"):
            return run_ocr_test_key(event["run_ocr_test_key"])
        if event.get("run_rmr_test_key"):
            return run_rmr_test_key(event["run_rmr_test_key"])
        if event.get("run_rmr_regenerate"):
            return run_rmr_regenerate(event["run_rmr_regenerate"])
        if event.get("run_rmr_send"):
            return run_rmr_send(event["run_rmr_send"], to=event.get("to"))
        if event.get("run_kpi_cleanup"):
            return run_kpi_cleanup(dry_run=bool(event.get("dry_run")))
        if event.get("run_catchup"):
            return run_catchup(event["run_catchup"], dry_run=bool(event.get("dry_run")))

        # ---- live path: SES receipt-rule invocation ----
        record = extract_ses_record(event)
        if not record:
            return response(200, {"message": "No SES record in event; ignored"})

        mail = record["ses"]["mail"]
        message_id = mail["messageId"]
        headers = mail.get("commonHeaders", {}) or {}
        sender = normalize_value(headers.get("from", mail.get("source", "")))
        to = normalize_value(headers.get("to", mail.get("destination", "")))
        subject = headers.get("subject", "") or ""
        received_date = infer_report_date(mail.get("timestamp") or headers.get("date") or "")

        print(f"SES inbound messageId={message_id} subject={subject!r} from={sender!r}")
        return route_message(message_id, sender, to, subject, received_date)
    except Exception as exc:
        print(f"ERROR: Lambda run failed: {exc}")
        raise


def extract_ses_record(event):
    """Return the first SES record, or None if this isn't an SES event."""
    for record in event.get("Records", []) or []:
        if record.get("eventSource") == "aws:ses" and record.get("ses"):
            return record
    return None


def route_message(message_id, sender, to, subject, received_date):
    """Read the raw email from S3 and dispatch to the KPI, RMR or PDF path."""
    raw_key = f"{RAW_PREFIX}{message_id}"
    raw_bytes = fetch_raw_email(raw_key)
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)

    # Prefer the parsed MIME headers when SES commonHeaders were empty.
    sender = sender or normalize_value(msg.get("From", ""))
    subject = subject or (msg.get("Subject", "") or "")
    return dispatch(message_id, msg, sender, to, subject, received_date)


def dispatch(message_id, msg, sender, to, subject, received_date):
    """Single routing point, shared by the live path and the reprocess modes."""
    if is_hubspot_kpi_email(sender, subject):
        return process_hubspot_kpi_email(message_id, msg, sender, to, subject, received_date)

    attachments = extract_pdf_attachments(msg)
    if is_rmr_pricing_email(sender, subject, [name for name, _cid, _b in attachments]):
        return process_rmr_pricing_email(
            message_id, attachments, sender, to, subject, received_date
        )
    return process_pdf_email(message_id, msg, sender, to, subject, received_date)


def fetch_raw_email(raw_key, attempts=5, delay=1.0):
    """
    GET the raw MIME the SES S3 action just wrote.

    SES runs the S3 action before the Lambda action, and S3 read-after-write is
    strongly consistent, so this normally succeeds first try.  The small retry
    only guards against the rare case where the async Lambda fires a beat early.
    """
    last_exc = None
    for i in range(attempts):
        try:
            return s3.get_object(Bucket=S3_BUCKET, Key=raw_key)["Body"].read()
        except s3.exceptions.NoSuchKey as exc:
            last_exc = exc
            print(f"  raw email not present yet at {raw_key} (try {i + 1}/{attempts})")
            time.sleep(delay)
    raise RuntimeError(f"Raw email never appeared at s3://{S3_BUCKET}/{raw_key}: {last_exc}")


# =====================================================================
# MIME extraction
# =====================================================================
def iter_parts(msg):
    """Yield leaf (non-multipart) parts with a decoded payload."""
    for part in msg.walk():
        if part.is_multipart():
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        yield part, payload


def extract_images(msg):
    """Return [(filename, bytes)] for every image part (inline cid charts included)."""
    images = []
    index = 0
    for part, payload in iter_parts(msg):
        if not part.get_content_type().startswith("image/"):
            continue
        name = part.get_filename() or f"inline_{index}.png"
        images.append((sanitize_filename(name), payload))
        index += 1
    return images


def extract_pdf_attachments(msg):
    """Return [(filename, content_id, bytes)] for every PDF part."""
    pdfs = []
    for part, payload in iter_parts(msg):
        filename = part.get_filename() or ""
        content_type = part.get_content_type()
        if filename.lower().endswith(".pdf") or content_type == "application/pdf":
            content_id = (part.get("Content-ID") or "").strip("<>")
            pdfs.append((sanitize_filename(filename or "attachment.pdf"), content_id, payload))
    return pdfs


# =====================================================================
# River Metals (RMR) price-sheet path
# =====================================================================
def is_rmr_pricing_email(sender, subject, filenames=None):
    """
    Identify the weekly River Metals NF price sheet.

    Matched on the sender hint (set RMR_SENDER_HINT once the sending domain is
    known) OR on "River Metals" / "NF Pricing" appearing in the subject or in an
    attachment filename.  Deliberately narrow so a ScrapRight PDF can never be
    mistaken for a price sheet.
    """
    sender_l = (sender or "").lower()
    if RMR_SENDER_HINT and RMR_SENDER_HINT in sender_l:
        return True
    hay = " ".join([subject or ""] + list(filenames or [])).lower().replace("_", " ")
    if re.search(r"river\s*metals", hay):
        return True
    if re.search(r"\bnf\s*pricing\b", hay):
        return True
    return False


def extract_rmr_date(text, filename, received_date):
    """
    Date for the RMR folder, in priority order:
      1. the M/D/YYYY printed at the top of the sheet (e.g. 8/26/2026),
      2. a M.D.YY / M.D.YYYY stamp in the filename (e.g. "... 8.26.26.pdf"),
      3. the email received date.
    Returns (YYYY-MM-DD, source).
    """
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text or "")
    if match:
        month, day, year = (int(match.group(i)) for i in (1, 2, 3))
        try:
            return date(year, month, day).strftime("%Y-%m-%d"), "pdf_header"
        except ValueError:
            pass

    match = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})(?=\D*$)", filename or "")
    if match:
        month, day, year = (int(match.group(i)) for i in (1, 2, 3))
        if year < 100:
            year += 2000
        try:
            return date(year, month, day).strftime("%Y-%m-%d"), "filename"
        except ValueError:
            pass

    return received_date, "email_received"


def normalize_rmr_filename(filename):
    """
    Trim River Metals' per-send token off the price-sheet filename.

    The vendor mails the sheet as e.g.

        River_Metals_NF_Pricing_8.26.26_f_mtw1d3.pdf

    where everything after the M.D.YY stamp changes on every send.  Keeping the
    name only through the date gives a predictable S3 key and a clean
    attachment name:

        River_Metals_NF_Pricing_8.26.26.pdf

    A name with no date stamp is returned unchanged.  Trimming also repairs
    extract_rmr_date's filename fallback: its stamp regex requires no digits
    after the date, and the token usually contains some.
    """
    base, dot, ext = filename.rpartition(".")
    if not dot:
        base, ext = filename, ""
    stamps = list(re.finditer(r"\d{1,2}\.\d{1,2}\.\d{2,4}", base))
    if not stamps:
        return filename
    trimmed = base[: stamps[-1].end()].rstrip("._-")
    if not trimmed:
        return filename
    return f"{trimmed}.{ext}" if ext else trimmed


def rmr_discount_name(filename, pct):
    """River_Metals_NF_Pricing_8.26.26.pdf -> River_Metals_NF_Pricing_8.26.26_20.pdf"""
    return insert_suffix(filename, f"{pct:g}")


def discount_price_sheet(pdf_bytes, pct):
    """
    Return (new_pdf_bytes, prices_changed) for a copy of the sheet with every
    product price reduced by `pct` percent.

    The original PDF is edited in place: each price span is redacted and the new
    value drawn right-aligned to the same right edge at the same font size, so
    the sheet is visually identical apart from the numbers.  Header market
    quotes (above RMR_TABLE_TOP_Y) and "call for price" rows are untouched.
    """
    plain_price = re.compile(r"^\d+\.\d{2}$")
    dollar_price = re.compile(r"^\$\d+\.\d{2}$")
    factor = (100.0 - float(pct)) / 100.0

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = 0
    for page in doc:
        edits = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # not a text block
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if span["bbox"][1] < RMR_TABLE_TOP_Y:
                        continue  # Comex / LME market quotes
                    if not (plain_price.match(text) or dollar_price.match(text)):
                        continue  # blanks, codes, "call for price"
                    prefix = "$" if text.startswith("$") else ""
                    value = float(text.lstrip("$"))
                    new_value = round(value * factor + 1e-9, 2)
                    edits.append((span, f"{prefix}{new_value:.2f}"))

        for span, _new in edits:
            page.add_redact_annot(fitz.Rect(span["bbox"]))
        # PDF_REDACT_IMAGE_NONE keeps the logo/artwork intact.
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        for span, new_text in edits:
            x0, y0, x1, y1 = span["bbox"]
            size = span["size"]
            width = fitz.get_text_length(new_text, fontname="helv", fontsize=size)
            page.insert_text(
                (x1 - width, y1 - (y1 - y0) * 0.22),
                new_text,
                fontname="helv",
                fontsize=size,
                color=(0, 0, 0),
            )
        total += len(edits)

    # Footnote only.  The old red "LESS n%" banner at (33, 62) was removed
    # 2026-09-10 - the discount is already stated in the filename and here.
    first = doc[0]
    first.insert_text(
        (33, 664),
        f"All prices above reflect a {pct:g}% discount from the published River Metals sheet.",
        fontname="helv",
        fontsize=8,
        color=(0.35, 0.35, 0.35),
    )
    return doc.tobytes(), total


def process_rmr_pricing_email(message_id, attachments, sender, to, subject, received_date):
    """Store the RMR price sheet, then generate + store the discounted copies."""
    print(f"Detected River Metals price-sheet email messageId={message_id}")
    processed = []
    warnings = []

    for raw_filename, content_id, pdf_bytes in attachments:
        safe_filename = normalize_rmr_filename(raw_filename)
        text = extract_pdf_text(pdf_bytes)
        report_date, date_source = extract_rmr_date(text, safe_filename, received_date)
        if date_source == "email_received":
            warnings.append(f"{safe_filename}: no sheet date found; used received date")
        year, month, day = report_date.split("-")

        stored_filename = safe_filename
        original_key = join_key(RMR_PREFIX, year, month, day, stored_filename)
        # The stored name is deterministic now that the per-send token is
        # trimmed, so a reprocess of the same email would fork a suffixed
        # duplicate.  Only suffix when a DIFFERENT file already holds the name.
        existing_etag = s3_object_etag(original_key)
        if existing_etag and existing_etag != hashlib.md5(pdf_bytes).hexdigest():
            stored_filename = insert_suffix(
                safe_filename, (content_id or message_id or "dup")[:8]
            )
            original_key = join_key(RMR_PREFIX, year, month, day, stored_filename)
            print(f"Name collision at {report_date}; storing as {stored_filename}")

        s3.put_object(
            Bucket=S3_BUCKET, Key=original_key, Body=pdf_bytes, ContentType="application/pdf"
        )
        print(f"Saved original price sheet to s3://{S3_BUCKET}/{original_key}")

        generated = []
        for pct in RMR_DISCOUNTS:
            try:
                new_bytes, changed = discount_price_sheet(pdf_bytes, pct)
            except Exception as exc:
                print(f"  discount {pct}% failed for {stored_filename}: {exc}")
                warnings.append(f"{stored_filename}: {pct:g}% sheet failed: {exc}")
                continue
            if not changed:
                warnings.append(f"{stored_filename}: no prices found for the {pct:g}% sheet")
            discounted_name = rmr_discount_name(stored_filename, pct)
            discounted_key = join_key(RMR_PREFIX, year, month, day, discounted_name)
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=discounted_key,
                Body=new_bytes,
                ContentType="application/pdf",
            )
            print(
                f"  wrote {pct:g}% sheet ({changed} prices) to s3://{S3_BUCKET}/{discounted_key}"
            )
            generated.append(
                {"discount_pct": pct, "key": discounted_key, "prices_changed": changed}
            )

        debug_text_key = join_key(DEBUG_TEXT_PREFIX, year, month, day, f"{stored_filename}.txt")
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=debug_text_key,
            Body=text.encode("utf-8", errors="replace"),
            ContentType="text/plain",
        )

        processed.append(
            {
                "filename": stored_filename,
                "report_date": report_date,
                "date_source": date_source,
                "original_key": original_key,
                "discounted": generated,
                "debug_text_key": debug_text_key,
            }
        )

    # ---- forward the sheets on ----
    # The PDFs are already safely in S3 at this point, so a send failure is
    # reported but never re-raised: we do not want a mail problem to look like
    # an ingest failure or to trigger a Lambda retry that re-writes objects.
    email_result = None
    if not processed:
        pass
    elif not RMR_AUTO_SEND:
        email_result = {"status": "skipped", "reason": "RMR_AUTO_SEND is off"}
        print("Auto-send disabled; sheets stored but not emailed")
    elif not RMR_MAIL_TO:
        email_result = {"status": "skipped", "reason": "RMR_MAIL_TO is not set"}
        warnings.append("sheets stored but not emailed: RMR_MAIL_TO is not set")
        print("RMR_MAIL_TO is empty; sheets stored but not emailed")
    else:
        try:
            email_result = send_rmr_email(processed, RMR_MAIL_TO)
        except Exception as exc:
            print(f"ERROR: RMR send failed: {exc}")
            email_result = {"status": "failed", "error": str(exc)}
            warnings.append(f"email send failed: {exc}")

    status_key = write_status(
        {
            "status": "success" if not warnings else "success_with_warnings",
            "message_id": message_id,
            "sender": sender,
            "to": to,
            "subject": subject,
            "received_date": received_date,
            "report_type": "rmr_price_sheet",
            "discounts": RMR_DISCOUNTS,
            "attachments_seen": len(attachments),
            "processed_pdf_count": len(processed),
            "processed_files": processed,
            "email": email_result,
            "warnings": warnings,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        received_date,
        message_id,
    )
    return response(
        200,
        {
            "message": "Processed River Metals price sheet",
            "message_id": message_id,
            "processed_pdf_count": len(processed),
            "processed_files": processed,
            "email": email_result,
            "warnings": warnings,
            "status_key": status_key,
        },
    )


# ---------------------------------------------------------------------
# RMR outbound email
# ---------------------------------------------------------------------
def build_rmr_message(processed, recipients):
    """
    Build the outbound MIME message carrying the price sheets.

    `processed` is the list produced by process_rmr_pricing_email (one entry per
    source PDF, each with its original_key and generated discounted sheets).
    Attachments are streamed back out of S3 rather than kept in memory from the
    generation step, so this also works for a manual re-send later.
    """
    from email.message import EmailMessage

    report_date = processed[0].get("report_date", "")
    msg = EmailMessage()
    msg["From"] = RMR_MAIL_FROM
    msg["To"] = ", ".join(recipients)
    if RMR_MAIL_CC:
        msg["Cc"] = ", ".join(RMR_MAIL_CC)
    if RMR_MAIL_REPLY_TO:
        msg["Reply-To"] = ", ".join(RMR_MAIL_REPLY_TO)
    pct_list = " and ".join(f"{p:g}%" for p in RMR_DISCOUNTS)
    msg["Subject"] = f"River Metals NF Pricing {report_date} - {pct_list} sheets"

    lines = [
        f"River Metals non-ferrous pricing for {report_date}.",
        "",
        f"Attached are the discounted sheets ({pct_list} off the published prices).",
        "",
    ]
    attachments = []
    for entry in processed:
        if RMR_ATTACH_ORIGINAL and entry.get("original_key"):
            attachments.append(entry["original_key"])
        for sheet in entry.get("discounted", []):
            if sheet.get("key"):
                attachments.append(sheet["key"])
    for key in attachments:
        lines.append(f"  - {key.rsplit('/', 1)[-1]}")
    lines += [
        "",
        "Sent automatically by the Miami Valley reports pipeline.",
    ]
    msg.set_content("\n".join(lines))

    for key in attachments:
        data = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        msg.add_attachment(
            data, maintype="application", subtype="pdf", filename=key.rsplit("/", 1)[-1]
        )
    return msg, attachments


def send_rmr_email(processed, recipients):
    """Send the price sheets via SES v2.  Returns a result dict for the status record."""
    msg, attached = build_rmr_message(processed, recipients)
    raw = msg.as_bytes()
    if len(raw) > RMR_MAX_MESSAGE_BYTES:
        raise RuntimeError(
            f"message is {len(raw)} bytes, over the {RMR_MAX_MESSAGE_BYTES}-byte limit"
        )

    destination = {"ToAddresses": list(recipients)}
    if RMR_MAIL_CC:
        destination["CcAddresses"] = list(RMR_MAIL_CC)

    result = sesv2.send_email(
        FromEmailAddress=RMR_MAIL_FROM,
        Destination=destination,
        Content={"Raw": {"Data": raw}},
    )
    message_id = result.get("MessageId")
    print(
        f"Emailed {len(attached)} sheet(s) to {', '.join(recipients)} "
        f"({len(raw)} bytes, SES MessageId={message_id})"
    )
    return {
        "status": "sent",
        "ses_message_id": message_id,
        "from": RMR_MAIL_FROM,
        "to": list(recipients),
        "cc": list(RMR_MAIL_CC),
        "reply_to": list(RMR_MAIL_REPLY_TO),
        "subject": msg["Subject"],
        "attachments": attached,
        "bytes": len(raw),
    }


# =====================================================================
# HubSpot Weekly KPI path
# =====================================================================
def is_hubspot_kpi_email(sender, subject):
    sender_match = "hubspot.com" in (sender or "").lower()
    subject_match = "kpi" in (subject or "").lower() or "weekly" in (subject or "").lower()
    return sender_match and subject_match


def process_hubspot_kpi_email(message_id, msg, sender, to, subject, received_date):
    print(f"Detected HubSpot Weekly KPI email messageId={message_id}")
    period_start, period_end = previous_week_bounds(
        datetime.strptime(received_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    )

    if KPI_SOURCE == "hubspot_api":
        metrics = collect_metrics_from_hubspot(period_start, period_end)
        source = "hubspot_crm_api"
    else:
        images = extract_images(msg)
        print(f"Found {len(images)} inline image(s) in messageId={message_id}")
        metrics = collect_metrics_from_images(images, received_date)
        source = "email_image_ocr"

    rows = []
    warnings = []
    for metric_key, label, _pattern, _obj, _prop in KPI_METRICS:
        found = metrics.get(metric_key, {})
        count = found.get("count", "")
        if count == "":
            warnings.append(f"{label}: no value extracted")
        if found.get("low_confidence"):
            warnings.append(
                f"{label}: OCR confidence {found.get('confidence')} below {MIN_CONFIDENCE}"
            )
        rows.append(
            {
                "week_ending_date": received_date,
                "metric": label,
                "metric_key": metric_key,
                "count": count,
                "change_pct": found.get("change_pct", ""),
                "direction": found.get("direction", ""),
            }
        )

    if not any(row["count"] for row in rows):
        print(f"WARNING: no KPI values extracted for messageId={message_id}. No CSV written.")
        write_status(
            {
                "status": "no_metrics_parsed",
                "message_id": message_id,
                "sender": sender,
                "to": to,
                "subject": subject,
                "received_date": received_date,
                "report_type": "hubspot_weekly_kpi",
                "data_source": source,
                "warnings": warnings,
                "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            received_date,
            message_id,
        )
        return response(
            200,
            {
                "message": "No KPI values extracted; no CSV written",
                "message_id": message_id,
                "warnings": warnings,
            },
        )

    r_year, r_month, r_day = received_date.split("-")
    csv_key = join_key(
        MAIN_PREFIX, r_year, r_month, r_day, f"hubspot_weekly_kpis_{received_date}.csv"
    )
    write_csv_to_s3(csv_key, rows)
    print(f"Wrote HubSpot KPI CSV to s3://{S3_BUCKET}/{csv_key}")

    write_status(
        {
            "status": "success" if not warnings else "success_with_warnings",
            "message_id": message_id,
            "sender": sender,
            "to": to,
            "subject": subject,
            "received_date": received_date,
            "report_type": "hubspot_weekly_kpi",
            "data_source": source,
            "period_start": period_start.strftime("%Y-%m-%d"),
            "period_end": period_end.strftime("%Y-%m-%d"),
            "csv_key": csv_key,
            "rows_written": len(rows),
            "metrics": metrics,
            "warnings": warnings,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        received_date,
        message_id,
    )
    return response(
        200,
        {
            "message": "Processed HubSpot Weekly KPI email",
            "message_id": message_id,
            "csv_key": csv_key,
            "rows_written": len(rows),
            "warnings": warnings,
            "metrics": {k: v.get("count") for k, v in metrics.items()},
        },
    )


# ---------------------------------------------------------------------
# OCR extraction
# ---------------------------------------------------------------------
def collect_metrics_from_images(images, received_date):
    """OCR each chart image and map it to a metric by the title inside the image."""
    metrics = {}
    for filename, image_bytes in images:
        try:
            lines, confidence = ocr_lines(image_bytes)
        except Exception as exc:
            print(f"Textract failed on {filename}: {exc}")
            continue

        joined = " ".join(lines)
        print(f"  OCR {filename}: {joined[:200]!r}")

        matched_key = None
        matched_label = None
        for metric_key, label, pattern, _obj, _prop in KPI_METRICS:
            if re.search(pattern, joined, re.IGNORECASE):
                matched_key = metric_key
                matched_label = label
                break
        if not matched_key:
            print(f"  {filename}: no KPI title matched; skipping")
            continue

        count = extract_count(lines)
        change_pct = extract_percent(joined)
        direction = arrow_direction(image_bytes)
        if not direction:
            direction = direction_from_previous_week(matched_key, count, received_date)

        metrics[matched_key] = {
            "count": count,
            "change_pct": change_pct,
            "direction": direction,
            "confidence": round(confidence, 1),
            "low_confidence": confidence < MIN_CONFIDENCE,
            "image": filename,
        }
        print(
            f"  {matched_label}: count={count!r}, change={change_pct!r}, "
            f"direction={direction!r}, confidence={confidence:.1f}"
        )
    return metrics


def ocr_lines(image_bytes):
    """Run Textract and return (list_of_line_strings, mean_confidence)."""
    result = textract.detect_document_text(Document={"Bytes": image_bytes})
    lines = []
    confidences = []
    for block in result.get("Blocks", []):
        if block.get("BlockType") == "LINE":
            lines.append(block.get("Text", ""))
            confidences.append(block.get("Confidence", 0.0))
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return lines, mean_confidence


def extract_count(lines):
    """The count is the standalone integer following the '(COUNT) X' caption."""
    for index, line in enumerate(lines):
        if re.search(r"\(\s*COUNT\s*\)", line, re.IGNORECASE):
            for candidate in lines[index + 1: index + 4]:
                cleaned = candidate.strip().replace(",", "")
                if re.fullmatch(r"\d+", cleaned):
                    return cleaned
    for line in lines:
        cleaned = line.strip().replace(",", "")
        if re.fullmatch(r"\d+", cleaned):
            return cleaned
    return ""


def extract_percent(joined):
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", joined)
    return match.group(1) if match else ""


def arrow_direction(image_bytes):
    """Classify the trend arrow by colour: teal up, red down, else blank."""
    try:
        pixels = load_rgb_pixels(image_bytes)
    except Exception as exc:
        print(f"  arrow colour detection unavailable: {exc}")
        return ""

    red = 0
    teal = 0
    for r, g, b in pixels:
        if r > 150 and r - g > 60 and r - b > 60:
            red += 1
        elif g > 120 and g - r > 60:
            teal += 1

    print(f"  arrow pixels: red={red}, teal={teal}")
    if red == 0 and teal == 0:
        return ""
    if red > teal * 2:
        return "down"
    if teal > red * 2:
        return "up"
    return ""


def load_rgb_pixels(image_bytes):
    """Yield (r, g, b) tuples using PyMuPDF (already in the layer for PDFs)."""
    try:
        pix = fitz.Pixmap(image_bytes)
    except Exception:
        doc = fitz.open(stream=image_bytes, filetype="png")
        pix = doc[0].get_pixmap()

    if pix.n >= 4:  # drop alpha / convert CMYK
        pix = fitz.Pixmap(fitz.csRGB, pix)

    samples = pix.samples
    channels = pix.n
    return [
        (samples[i], samples[i + 1], samples[i + 2])
        for i in range(0, len(samples) - channels + 1, channels)
    ]


def direction_from_previous_week(metric_key, count, received_date):
    """Fallback: compare against the count in the previous week's CSV."""
    if not count:
        return ""
    try:
        previous_date = (
            datetime.strptime(received_date, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")
        p_year, p_month, p_day = previous_date.split("-")
        key = join_key(
            MAIN_PREFIX, p_year, p_month, p_day, f"hubspot_weekly_kpis_{previous_date}.csv"
        )
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")
        for row in csv.DictReader(io.StringIO(body)):
            if row.get("metric_key") == metric_key and (row.get("count") or "").strip():
                prior = int(row["count"])
                current = int(count)
                if current > prior:
                    return "up"
                if current < prior:
                    return "down"
                return "flat"
    except Exception as exc:
        print(f"  no previous-week comparison for {metric_key}: {exc}")
    return ""


# ---------------------------------------------------------------------
# Optional HubSpot CRM API source (KPI_SOURCE=hubspot_api)
# ---------------------------------------------------------------------
def collect_metrics_from_hubspot(period_start, period_end):
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    if not HUBSPOT_TOKEN:
        raise RuntimeError("KPI_SOURCE=hubspot_api but HUBSPOT_TOKEN is not set")

    def hubspot_post(path, payload):
        req = Request(
            f"{HUBSPOT_API_BASE}{path}",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {HUBSPOT_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urlopen(req, timeout=30) as res:
                return json.loads(res.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"HubSpot HTTP {exc.code} for {path}: {exc.read().decode('utf-8', 'replace')}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"HubSpot connection error for {path}: {exc}") from exc

    def hubspot_count(object_type, date_property, start, end):
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": date_property,
                            "operator": "BETWEEN",
                            "value": str(to_epoch_ms(start)),
                            "highValue": str(to_epoch_ms(end)),
                        }
                    ]
                }
            ],
            "limit": 1,
            "properties": [date_property],
        }
        return int(hubspot_post(f"/crm/v3/objects/{object_type}/search", payload).get("total", 0))

    prior_start, prior_end = previous_week_bounds(period_start)
    metrics = {}
    for metric_key, label, _pattern, object_type, date_property in KPI_METRICS:
        count = hubspot_count(object_type, date_property, period_start, period_end)
        prior = hubspot_count(object_type, date_property, prior_start, prior_end)
        change_pct, direction = percent_change(prior, count)
        metrics[metric_key] = {
            "count": str(count),
            "change_pct": change_pct,
            "direction": direction,
            "confidence": 100.0,
            "low_confidence": False,
        }
        print(f"  {label}: count={count}, prior={prior}, change={change_pct!r}")
    return metrics


def percent_change(prior, current):
    if prior == 0:
        return ("0", "flat") if current == 0 else ("", "up")
    delta = (current - prior) / prior * 100.0
    direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    return f"{abs(delta):.1f}", direction


def to_epoch_ms(dt):
    return int(dt.timestamp() * 1000)


def previous_week_bounds(reference):
    """(Mon 00:00:00, Sun 23:59:59.999999) UTC of the week before the reference."""
    this_monday = (reference - timedelta(days=reference.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return this_monday - timedelta(days=7), this_monday - timedelta(microseconds=1)


# =====================================================================
# ScrapRight PDF path
# =====================================================================
def process_pdf_email(message_id, msg, sender, to, subject, received_date):
    r_year, r_month, r_day = received_date.split("-")
    attachments = extract_pdf_attachments(msg)
    print(f"Found {len(attachments)} PDF attachment(s) in messageId={message_id}")

    processed_files = []
    for safe_filename, content_id, pdf_bytes in attachments:
        text = extract_pdf_text(pdf_bytes)

        internal_date = extract_report_date(text)
        report_date = internal_date or received_date
        if not internal_date:
            print(
                f"WARNING: no report date inside {safe_filename}; using received_date {report_date}"
            )
        p_year, p_month, p_day = report_date.split("-")

        stored_filename = safe_filename
        pdf_key = join_key(RAW_PDF_PREFIX, p_year, p_month, p_day, stored_filename)
        if s3_object_exists(pdf_key):
            stored_filename = insert_suffix(safe_filename, (content_id or message_id or "dup")[:8])
            pdf_key = join_key(RAW_PDF_PREFIX, p_year, p_month, p_day, stored_filename)
            print(f"Name collision at {report_date}; storing as {stored_filename}")

        s3.put_object(Bucket=S3_BUCKET, Key=pdf_key, Body=pdf_bytes, ContentType="application/pdf")
        print(f"Saved PDF to s3://{S3_BUCKET}/{pdf_key}")

        debug_text_key = join_key(
            DEBUG_TEXT_PREFIX, p_year, p_month, p_day, f"{stored_filename}.txt"
        )
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=debug_text_key,
            Body=text.encode("utf-8", errors="replace"),
            ContentType="text/plain",
        )

        processed_files.append(
            {
                "filename": stored_filename,
                "report_date": report_date,
                "date_source": "pdf" if internal_date else "email_received",
                "pdf_key": pdf_key,
                "debug_text_key": debug_text_key,
            }
        )

    status_key = write_status(
        {
            "status": "success",
            "message_id": message_id,
            "sender": sender,
            "to": to,
            "subject": subject,
            "received_date": received_date,
            "report_type": "scrapright_pdf",
            "attachments_seen": len(attachments),
            "processed_pdf_count": len(processed_files),
            "processed_files": processed_files,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        received_date,
        message_id,
    )
    return response(
        200,
        {
            "message": "Processed inbound email",
            "message_id": message_id,
            "processed_pdf_count": len(processed_files),
            "status_key": status_key,
        },
    )


def extract_report_date(text):
    for pattern in [
        r"For Dates From\s+(\d{1,2})/(\d{1,2})/(\d{4})",
        r"\bDate\s*:\s*(\d{1,2})/(\d{1,2})/(\d{4})",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            month, day, year = (int(match.group(i)) for i in (1, 2, 3))
            try:
                return date(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return ""


def extract_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)


# =====================================================================
# Operational modes
# =====================================================================
def reprocess_raw_key(raw_key):
    """Reparse one raw email already stored in S3 (key like raw/<messageId>)."""
    message_id = raw_key.split("/")[-1]
    raw_bytes = s3.get_object(Bucket=S3_BUCKET, Key=raw_key)["Body"].read()
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)
    sender = normalize_value(msg.get("From", ""))
    to = normalize_value(msg.get("To", ""))
    subject = msg.get("Subject", "") or ""
    received_date = infer_report_date(msg.get("Date", ""))
    return dispatch(message_id, msg, sender, to, subject, received_date)


def run_rmr_test_key(raw_key):
    """
    Dry-run the RMR path on a stored raw email: report whether it would be
    recognised as a price sheet, what date/folder it would use, and a sample of
    the discounted values.  Writes nothing.
    """
    raw_bytes = s3.get_object(Bucket=S3_BUCKET, Key=raw_key)["Body"].read()
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)
    sender = normalize_value(msg.get("From", ""))
    subject = msg.get("Subject", "") or ""
    received_date = infer_report_date(msg.get("Date", ""))
    attachments = extract_pdf_attachments(msg)
    names = [name for name, _cid, _b in attachments]
    recognised = is_rmr_pricing_email(sender, subject, names)

    report = []
    for raw_filename, _cid, pdf_bytes in attachments:
        safe_filename = normalize_rmr_filename(raw_filename)
        text = extract_pdf_text(pdf_bytes)
        report_date, date_source = extract_rmr_date(text, safe_filename, received_date)
        entry = {
            "filename": safe_filename,
            "source_filename": raw_filename,
            "report_date": report_date,
            "date_source": date_source,
            "would_store_at": join_key(RMR_PREFIX, *report_date.split("-"), safe_filename),
            "sheets": [],
        }
        for pct in RMR_DISCOUNTS:
            try:
                _new_bytes, changed = discount_price_sheet(pdf_bytes, pct)
                entry["sheets"].append({"discount_pct": pct, "prices_changed": changed})
            except Exception as exc:
                entry["sheets"].append({"discount_pct": pct, "error": str(exc)})
        report.append(entry)

    return response(
        200,
        {
            "message": "RMR dry run (nothing written)",
            "recognised_as_price_sheet": recognised,
            "sender": sender,
            "subject": subject,
            "attachments": names,
            "discounts": RMR_DISCOUNTS,
            "report": report,
        },
    )


def run_rmr_regenerate(original_key):
    """
    Rebuild the discounted sheets from an original already stored under RMR/.
    Use after changing RMR_DISCOUNTS, or to repair a partial run.
    """
    pdf_bytes = s3.get_object(Bucket=S3_BUCKET, Key=original_key)["Body"].read()
    folder, _slash, filename = original_key.rpartition("/")
    generated = []
    for pct in RMR_DISCOUNTS:
        new_bytes, changed = discount_price_sheet(pdf_bytes, pct)
        key = join_key(folder, rmr_discount_name(filename, pct))
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=new_bytes, ContentType="application/pdf")
        print(f"Regenerated {pct:g}% sheet ({changed} prices) at s3://{S3_BUCKET}/{key}")
        generated.append({"discount_pct": pct, "key": key, "prices_changed": changed})
    return response(
        200,
        {
            "message": "Regenerated discounted price sheets",
            "original_key": original_key,
            "generated": generated,
        },
    )


def run_rmr_send(selector, to=None):
    """
    Manually email the sheets for an already-stored original.  Useful for a
    re-send, for testing a new recipient, or after RMR_AUTO_SEND was off.

        {"run_rmr_send": "RMR/2026/08/26/River_Metals_NF_Pricing_8.26.26.pdf"}
        {"run_rmr_send": "RMR/2026/08/26/....pdf", "to": "someone@example.com"}
        {"run_rmr_send": {"key": "RMR/.../x.pdf", "to": ["a@b.com", "c@d.com"]}}

    Recipients fall back to RMR_MAIL_TO when none are given.
    """
    if isinstance(selector, dict):
        original_key = selector.get("key") or selector.get("original_key") or ""
        to = selector.get("to", to)
    else:
        original_key = str(selector)
    if not original_key:
        return response(400, {"error": "run_rmr_send needs the original PDF key"})

    if isinstance(to, str):
        recipients = [a.strip() for a in to.split(",") if a.strip()]
    elif isinstance(to, list):
        recipients = [str(a).strip() for a in to if str(a).strip()]
    else:
        recipients = list(RMR_MAIL_TO)
    if not recipients:
        return response(400, {"error": "no recipients: pass \"to\" or set RMR_MAIL_TO"})

    folder, _slash, filename = original_key.rpartition("/")
    match = re.search(r"(\d{4})/(\d{2})/(\d{2})", folder)
    report_date = "-".join(match.groups()) if match else ""

    discounted = []
    missing = []
    for pct in RMR_DISCOUNTS:
        key = join_key(folder, rmr_discount_name(filename, pct))
        if s3_object_exists(key):
            discounted.append({"discount_pct": pct, "key": key})
        else:
            missing.append(key)
    if missing:
        print(f"  missing discounted sheet(s): {missing} — run_rmr_regenerate rebuilds them")

    processed = [
        {
            "filename": filename,
            "report_date": report_date,
            "original_key": original_key,
            "discounted": discounted,
        }
    ]
    result = send_rmr_email(processed, recipients)
    return response(
        200,
        {
            "message": "Sent River Metals price sheets",
            "original_key": original_key,
            "missing_sheets": missing,
            "email": result,
        },
    )


def run_ocr_test_key(raw_key):
    """OCR the images in a stored email and report readings, writing nothing."""
    raw_bytes = s3.get_object(Bucket=S3_BUCKET, Key=raw_key)["Body"].read()
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)
    images = extract_images(msg)
    report = []
    for filename, image_bytes in images:
        try:
            lines, confidence = ocr_lines(image_bytes)
        except Exception as exc:
            report.append({"image": filename, "error": str(exc)})
            continue
        joined = " ".join(lines)
        matched = ""
        for metric_key, _label, pattern, _obj, _prop in KPI_METRICS:
            if re.search(pattern, joined, re.IGNORECASE):
                matched = metric_key
                break
        report.append(
            {
                "image": filename,
                "bytes": len(image_bytes),
                "matched_metric": matched,
                "count": extract_count(lines),
                "change_pct": extract_percent(joined),
                "direction": arrow_direction(image_bytes),
                "confidence": round(confidence, 1),
                "lines": lines,
            }
        )
    return response(
        200,
        {"message": "OCR test complete (nothing written)", "images": len(images), "report": report},
    )


def run_catchup(selector, dry_run=False):
    """
    Reprocess every raw email under RAW_PREFIX within a date window.

    selector:
        "YYYY-MM-DD"                       -> from that date through today
        {"since": "...", "until": "..."}   -> explicit window, both inclusive
    """
    if isinstance(selector, dict):
        since = selector.get("since", "")
        until = selector.get("until", "") or date.today().strftime("%Y-%m-%d")
    else:
        since = str(selector)
        until = date.today().strftime("%Y-%m-%d")

    targets = []
    for key in list_keys(RAW_PREFIX):
        if key.endswith("/"):
            continue
        try:
            raw_bytes = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)
        except Exception as exc:
            print(f"  skipping unreadable raw email {key}: {exc}")
            continue
        received_date = infer_report_date(msg.get("Date", ""))
        if since and received_date < since:
            continue
        if until and received_date > until:
            continue
        targets.append(
            {
                "raw_key": key,
                "received_date": received_date,
                "subject": msg.get("Subject", "") or "",
            }
        )

    targets.sort(key=lambda item: item.get("received_date") or "")
    print(f"Catch-up window {since}..{until}: {len(targets)} email(s)")
    for t in targets:
        print(f"  {t['received_date']}  {t['raw_key']}  {t['subject']!r}")

    if dry_run:
        return response(
            200,
            {
                "message": "Catch-up dry run — nothing processed",
                "dry_run": True,
                "window": {"since": since, "until": until},
                "target_count": len(targets),
                "targets": targets,
            },
        )

    results = []
    for t in targets:
        try:
            body = json.loads(reprocess_raw_key(t["raw_key"])["body"])
            results.append(
                {"raw_key": t["raw_key"], "received_date": t["received_date"], "result": body}
            )
        except Exception as exc:
            print(f"Catch-up failed for {t['raw_key']}: {exc}")
            results.append({"raw_key": t["raw_key"], "error": str(exc)})

    succeeded = sum(1 for r in results if not r.get("error"))
    return response(
        200,
        {
            "message": "Catch-up complete",
            "window": {"since": since, "until": until},
            "attempted": len(results),
            "succeeded": succeeded,
            "failed": len(results) - succeeded,
            "results": results,
        },
    )


def run_kpi_cleanup(dry_run=False):
    deleted, kept = [], []
    for key in list_keys(f"{MAIN_PREFIX}/" if MAIN_PREFIX else ""):
        name = key.rsplit("/", 1)[-1]
        if not name.startswith("hubspot_weekly_kpis_") or not name.endswith(".csv"):
            continue
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode(
            "utf-8", errors="replace"
        )
        if csv_is_blank(body):
            if dry_run:
                print(f"[dry-run] would delete blank CSV: {key}")
            else:
                s3.delete_object(Bucket=S3_BUCKET, Key=key)
                print(f"Deleted blank CSV: {key}")
            deleted.append(key)
        else:
            kept.append(key)
    return response(
        200,
        {
            "message": "KPI cleanup dry run" if dry_run else "KPI cleanup complete",
            "dry_run": bool(dry_run),
            "deleted_count": len(deleted),
            "deleted": deleted,
            "kept_count": len(kept),
            "kept": kept,
        },
    )


def csv_is_blank(body):
    try:
        reader = csv.DictReader(io.StringIO(body))
        rows = list(reader)
    except Exception:
        return False
    if not rows:
        return True
    if "count" not in (reader.fieldnames or []):
        return any("message" in row for row in rows)
    return all(not (row.get("count") or "").strip() for row in rows)


# =====================================================================
# S3 helpers
# =====================================================================
def s3_object_exists(key):
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        if code in ("403", "AccessDenied", "Forbidden"):
            print(f"WARNING: head_object denied for {key}; assuming absent")
            return False
        raise


def s3_object_etag(key):
    """
    MD5 of a stored object, or "" if it is absent / unreadable.

    S3 returns the MD5 as the ETag for single-part puts, which is what every
    put_object in this function produces, so it can be compared directly
    against hashlib.md5(body).hexdigest() to tell "same file again" from
    "different file, same name".
    """
    try:
        etag = s3.head_object(Bucket=S3_BUCKET, Key=key).get("ETag", "")
    except s3.exceptions.ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound", "403", "AccessDenied", "Forbidden"):
            return ""
        raise
    return etag.strip('"')


def list_keys(prefix):
    token = None
    while True:
        kwargs = {"Bucket": S3_BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for item in page.get("Contents", []):
            yield item["Key"]
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")


def write_status(status, received_date, message_id):
    r_year, r_month, r_day = received_date.split("-")
    status_key = join_key(STATUS_PREFIX, r_year, r_month, r_day, f"{message_id}.json")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=status_key,
        Body=json.dumps(status, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Wrote status to s3://{S3_BUCKET}/{status_key}")
    return status_key


def write_csv_to_s3(key, rows):
    output = io.StringIO()
    if not rows:
        fieldnames = ["message"]
        rows = [{"message": "No rows parsed"}]
    else:
        fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    s3.put_object(
        Bucket=S3_BUCKET, Key=key, Body=output.getvalue().encode("utf-8"), ContentType="text/csv"
    )


# =====================================================================
# Shared helpers
# =====================================================================
def join_key(*parts):
    """Join S3 key segments, dropping empty ones so a blank prefix never yields '//'."""
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def infer_report_date(received_at):
    """Parse an ISO-8601 or RFC-2822 date string to YYYY-MM-DD (UTC), else today."""
    value = str(received_at or "")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            dt = None
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def insert_suffix(filename, suffix):
    base, dot, ext = filename.rpartition(".")
    return f"{base}_{suffix}.{ext}" if dot else f"{filename}_{suffix}"


def sanitize_filename(filename):
    filename = (filename or "").strip().replace(" ", "_")
    filename = re.sub(r"[^A-Za-z0-9._-]", "", filename)
    return filename or "attachment"


def normalize_value(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }
