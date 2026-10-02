from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from pathlib import Path
from urllib.parse import urljoin
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse
import boto3
import hashlib
import os
import re
import sys


BASE_URL = "https://mvmr.scrapright.com"
TIMEZONE = ZoneInfo("America/New_York")

VALID_TICKET_PREFIXES = {"MV"}

MAX_PAGES_TO_PROCESS = 9999
MAX_TICKETS_TO_PROCESS = 9999
RESET_EVERY_N_PAGES = 9999
S3_UPLOAD_WORKERS = 8

# Real Playwright timeouts (milliseconds). Previously these were 0 (disabled),
# which meant any missing/changed selector hung the script forever instead of
# failing fast — the instance never reached its stop step.
DEFAULT_TIMEOUT_MS = 60_000
NAV_TIMEOUT_MS = 120_000

SKIP_STATUSES = {"CANCEL", "VOID"}

MAIN_DIR = Path(__file__).resolve().parent
load_dotenv(MAIN_DIR / ".env")

SCRAPRIGHT_USER = os.getenv("SCRAPRIGHT_USER")
SCRAPRIGHT_PASS = os.getenv("SCRAPRIGHT_PASS")

S3_BUCKET_NAME = "mvmr-scrapright-images-ojanama"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

ALL_TICKETS_DIR = "all_by_ticket"

PAGINATION_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*of\s*(\d+)\s*$")

s3 = boto3.client("s3")


def build_ticket_pattern(prefixes: set[str]) -> re.Pattern:
    r"""
    Build a regex that matches only approved ticket prefixes followed by digits.
    Example: {"MV", "AB"} -> ^(?:MV|AB)\d+$
    """
    if not prefixes:
        raise ValueError("VALID_TICKET_PREFIXES cannot be empty.")

    cleaned_prefixes = sorted({p.strip().upper() for p in prefixes if p.strip()}, key=len, reverse=True)
    if not cleaned_prefixes:
        raise ValueError("VALID_TICKET_PREFIXES cannot be empty after cleaning.")

    pattern = r"^(?:" + "|".join(re.escape(prefix) for prefix in cleaned_prefixes) + r")\d+$"
    return re.compile(pattern)


TICKET_PATTERN = build_ticket_pattern(VALID_TICKET_PREFIXES)


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape ScrapRight ticket images to S3.")
    parser.add_argument(
        "--from-date",
        dest="from_date",
        help="Start date in MM/DD/YYYY format",
    )
    parser.add_argument(
        "--to-date",
        dest="to_date",
        help="End date in MM/DD/YYYY format",
    )
    parser.add_argument(
        "--mode",
        choices=["weekly", "manual"],
        default="weekly",
        help="weekly = previous full Sunday-Saturday week; manual = requires explicit dates",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days to include for weekly mode (default: 7)",
    )
    return parser.parse_args()


def parse_mmddyyyy(date_str: str):
    return datetime.strptime(date_str, "%m/%d/%Y").date()


def format_mmddyyyy(date_obj):
    return date_obj.strftime("%m/%d/%Y")


def resolve_date_range(args):
    if args.from_date or args.to_date:
        if not (args.from_date and args.to_date):
            raise ValueError("You must provide both --from-date and --to-date together.")
        start_date = parse_mmddyyyy(args.from_date)
        end_date = parse_mmddyyyy(args.to_date)
    elif args.mode == "weekly":
        today_local = datetime.now(TIMEZONE).date()

        days_since_sunday = (today_local.weekday() + 1) % 7
        current_week_sunday = today_local - timedelta(days=days_since_sunday)

        start_date = current_week_sunday - timedelta(days=7)
        end_date = current_week_sunday - timedelta(days=1)
    else:
        raise ValueError("Manual mode requires --from-date and --to-date.")

    if start_date > end_date:
        raise ValueError("Start date cannot be after end date.")

    return format_mmddyyyy(start_date), format_mmddyyyy(end_date)


def normalize_ticket(text: str) -> str:
    return text.strip().upper()


def looks_like_ticket(text: str) -> bool:
    ticket = normalize_ticket(text)
    return bool(TICKET_PATTERN.match(ticket))


def clean_label(label: str) -> str:
    label = label.strip()
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^A-Za-z0-9_]+", "", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "Unknown"


def extension_from_content_type(content_type: str) -> str:
    content_type = (content_type or "").lower()
    if "png" in content_type:
        return ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"


def compact_date(date_str: str) -> str:
    month, day, year = date_str.split("/")
    return f"{month.zfill(2)}{day.zfill(2)}{year}"


def choose_dataset_split(ticket_number: str, item_index: int, material_label: str, filename: str) -> str:
    key = f"{ticket_number}|{item_index}|{material_label}|{filename}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF

    if value < TRAIN_RATIO:
        return "train"
    if value < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


def build_s3_key(dataset_split: str, material_label: str, filename: str) -> str:
    safe_label = clean_label(material_label)
    return f"{dataset_split}/{safe_label}/{filename}"


def build_ticket_lookup_key(ticket_number: str, filename: str) -> str:
    return f"{ALL_TICKETS_DIR}/{ticket_number}/{filename}"


def upload_bytes_to_s3(
    image_bytes: bytes,
    dataset_split: str,
    material_label: str,
    ticket_number: str,
    filename: str,
    content_type: str,
) -> tuple[str, str]:
    dataset_s3_key = build_s3_key(dataset_split, material_label, filename)
    ticket_lookup_s3_key = build_ticket_lookup_key(ticket_number, filename)

    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=dataset_s3_key,
        Body=image_bytes,
        ContentType=content_type or "application/octet-stream",
    )

    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=ticket_lookup_s3_key,
        Body=image_bytes,
        ContentType=content_type or "application/octet-stream",
    )

    dataset_s3_uri = f"s3://{S3_BUCKET_NAME}/{dataset_s3_key}"
    ticket_lookup_s3_uri = f"s3://{S3_BUCKET_NAME}/{ticket_lookup_s3_key}"

    return dataset_s3_uri, ticket_lookup_s3_uri


def login(page):
    print("Logging into ScrapRight...")

    if not SCRAPRIGHT_USER or not SCRAPRIGHT_PASS:
        raise RuntimeError("Missing SCRAPRIGHT_USER or SCRAPRIGHT_PASS in MainDir/.env")

    page.goto(BASE_URL, wait_until="load")

    username_input = page.locator("input#tUsername")
    password_input = page.locator("input#tPassword")
    sign_in_button = page.locator('button[type="submit"]:has-text("sign in")')

    username_input.wait_for()
    password_input.wait_for()
    sign_in_button.wait_for()

    username_input.fill(SCRAPRIGHT_USER)
    password_input.fill(SCRAPRIGHT_PASS)
    sign_in_button.click()

    print("Login submitted.")


DEBUG_S3_PREFIX = "_debug"


def save_debug_snapshot(page, label: str):
    """
    Best-effort record of what the browser is showing right now.

    Prints the URL, title and visible text into the run log, saves a screenshot
    to MVMR/logs, and copies the screenshot to s3://<bucket>/_debug/ so it can
    be viewed without starting the instance. Never raises: a failed capture
    must not hide the original error.
    """
    stamp = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")

    try:
        print(f"[debug] URL:   {page.url}")
        print(f"[debug] Title: {page.title()}")
        visible_text = page.locator("body").inner_text(timeout=5000)
        print("[debug] Visible text (first 1500 chars):")
        print(visible_text[:1500])
    except Exception as e:
        print(f"[debug] Could not read page state: {e}")

    try:
        logs_dir = MAIN_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        screenshot_path = logs_dir / f"{label}_{stamp}.png"
        page.screenshot(path=str(screenshot_path), full_page=True, timeout=15000)
        print(f"[debug] Screenshot saved to {screenshot_path}")
    except Exception as e:
        print(f"[debug] Could not save screenshot: {e}")
        return

    try:
        s3_key = f"{DEBUG_S3_PREFIX}/{screenshot_path.name}"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=screenshot_path.read_bytes(),
            ContentType="image/png",
        )
        print(f"[debug] Screenshot copied to s3://{S3_BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"[debug] Could not copy screenshot to S3: {e}")


def wait_for_login_to_complete(page):
    print("Waiting for login to complete in the browser...")

    dashboard_link = page.locator('a.nav-link[href="/home"]')
    transactions_link = page.locator('a.nav-link[href="/transaction/ticket"]')

    try:
        dashboard_link.wait_for()
        transactions_link.wait_for()
    except PlaywrightTimeoutError:
        save_debug_snapshot(page, "login_fail")
        raise

    print("Login detected. Continuing automatically...")


def go_to_admin_ticket_review(page):
    print("Opening TRANSACTIONS menu...")

    transactions = page.locator('a.nav-link[href="/transaction/ticket"]')
    transactions.wait_for()
    transactions.hover()
    page.wait_for_timeout(1500)

    admin_review = page.locator('a.dropdown-item[href="/transaction/ticketreview"]')
    admin_review.wait_for()
    admin_review.click()

    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)


def set_search_type_to_date_range(page):
    print("Selecting Date Range...")

    page.locator("text=TICKET SEARCH").wait_for()

    dropdown = page.locator(
        'div.filter-option-inner-inner:has-text("Customer/license/ticket#/ref#")'
    )
    dropdown.wait_for()
    dropdown.click()
    page.wait_for_timeout(700)

    date_range_option = page.locator('span.text:has-text("Date Range")')
    date_range_option.wait_for()
    date_range_option.click()
    page.wait_for_timeout(800)

    print("Date Range selected.")


def fill_date_range(page, start_date: str, end_date: str):
    print(f"Filling date range: {start_date} -> {end_date}")

    compact_start = compact_date(start_date)
    compact_end = compact_date(end_date)

    start_box = page.locator('input#fromDate_0')
    end_box = page.locator('input#toDate_0')

    start_box.wait_for()
    end_box.wait_for()

    start_box.click(position={"x": 15, "y": 15})
    page.wait_for_timeout(200)
    start_box.fill("")
    start_box.type(compact_start, delay=80)

    page.wait_for_timeout(300)

    end_box.click(position={"x": 15, "y": 15})
    page.wait_for_timeout(200)
    end_box.fill("")
    end_box.type(compact_end, delay=80)

    page.wait_for_timeout(500)

    print(f"Start date entered: {compact_start}")
    print(f"End date entered:   {compact_end}")


def click_search(page):
    print("Clicking search...")

    buttons = page.locator("button:visible")
    for i in range(buttons.count()):
        btn = buttons.nth(i)
        try:
            title = (btn.get_attribute("title") or "").lower()
            aria = (btn.get_attribute("aria-label") or "").lower()
            if "search" in title or "search" in aria:
                btn.click()
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1500)
                return
        except Exception:
            pass

    buttons.first.click()
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)


def run_search(page, start_date: str, end_date: str):
    go_to_admin_ticket_review(page)
    set_search_type_to_date_range(page)
    fill_date_range(page, start_date, end_date)
    click_search(page)


def wait_for_results_table(page):
    print("Waiting for search results...")
    page.locator('th.fw-extrabold:has-text("Ticket#")').first.wait_for()
    page.wait_for_timeout(1000)


def read_current_page_ticket_numbers(page):
    print("Scraping ticket numbers from current page...")

    wait_for_results_table(page)

    tickets = []
    seen = set()

    rows = page.locator('tbody > tr[id^="r_"]:visible')
    row_count = rows.count()

    for i in range(row_count):
        row = rows.nth(i)
        cells = row.locator("td")

        if cells.count() < 4:
            continue

        try:
            first_cell = cells.nth(0).inner_text(timeout=3000).strip()
        except Exception:
            print(f"Skipping row {i+1}: could not read ticket cell")
            continue

        candidate = normalize_ticket(first_cell)
        if not looks_like_ticket(candidate):
            continue

        try:
            status_text = cells.nth(3).inner_text(timeout=3000).strip().upper()
        except Exception:
            status_text = ""

        if any(skip_status in status_text for skip_status in SKIP_STATUSES):
            print(f"Skipping ticket {candidate} due to status: {status_text}")
            continue

        if candidate not in seen:
            tickets.append(candidate)
            seen.add(candidate)

    print(f"Found {len(tickets)} ticket(s) on this page after status filtering: {tickets}")
    return tickets


def get_pagination_text(page):
    candidates = [
        'div:has-text("of")',
        'span:has-text("of")',
        'p:has-text("of")',
        'td:has-text("of")',
    ]

    for selector in candidates:
        locator = page.locator(selector + ":visible")
        count = locator.count()

        for i in range(count):
            try:
                text = locator.nth(i).inner_text().strip()
            except Exception:
                continue

            if PAGINATION_PATTERN.match(text):
                return text

    return None


def get_pagination_info(page):
    text = get_pagination_text(page)
    if not text:
        return None

    match = PAGINATION_PATTERN.match(text)
    if not match:
        return None

    start_num, end_num, total_num = map(int, match.groups())
    return {
        "text": text,
        "start": start_num,
        "end": end_num,
        "total": total_num,
    }


def click_next_page(page):
    old_info = get_pagination_info(page)
    if not old_info:
        print("No pagination text found.")
        return False

    if old_info["end"] >= old_info["total"]:
        print("Reached last page.")
        return False

    print(f"Pagination before click: {old_info['text']}")

    next_arrow = page.locator(
        'span.material-symbols-rounded.color-black.fs-sm:has-text("arrow_forward_ios"):visible'
    ).first

    next_arrow.wait_for()

    clicked = False
    for target in (next_arrow, next_arrow.locator("xpath=..")):
        try:
            target.click()
            clicked = True
            break
        except Exception:
            continue

    if not clicked:
        print("Could not click next-page arrow.")
        return False

    page.wait_for_timeout(2000)

    new_info = get_pagination_info(page)
    if new_info and new_info["text"] != old_info["text"]:
        print(f"Pagination after click: {new_info['text']}")
        return True

    print("Next page click did not change pagination.")
    return False


def jump_to_page(page, target_page: int, start_date: str, end_date: str):
    print(f"Re-establishing search state and jumping to page {target_page}...")

    page.goto(BASE_URL, wait_until="load")
    wait_for_login_to_complete(page)
    run_search(page, start_date, end_date)

    current_page = 1
    while current_page < target_page:
        moved = click_next_page(page)
        if not moved:
            raise RuntimeError(f"Could not advance to target page {target_page}. Stopped at page {current_page}.")
        current_page += 1

    print(f"Arrived at page {target_page}.")


def wait_for_ticket_header(page, ticket_number: str):
    header = page.locator(f'text=TICKET # {ticket_number}')
    header.wait_for()
    page.wait_for_timeout(500)


def click_ticket_row(page, ticket_number: str):
    print(f"Opening ticket {ticket_number}...")

    rows = page.locator('tbody > tr[id^="r_"]:visible')
    row_count = rows.count()

    for i in range(row_count):
        row = rows.nth(i)
        cells = row.locator("td")

        if cells.count() == 0:
            continue

        try:
            first_cell = cells.nth(0).inner_text(timeout=3000).strip()
        except Exception:
            continue

        if normalize_ticket(first_cell) == ticket_number:
            row.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            row.click()
            page.wait_for_timeout(1200)
            wait_for_ticket_header(page, ticket_number)
            return True

    print(f"Could not find clickable row for {ticket_number}.")
    return False


def wait_for_ticket_detail_or_no_images(page, ticket_number: str):
    print(f"Waiting for ticket detail for {ticket_number}...")

    body = page.locator("body")
    stable_checks = 0
    last_signature = None

    while True:
        page.wait_for_timeout(500)

        item_cards = page.locator("div.ticketItem:visible")
        card_count = item_cards.count()

        if card_count > 0:
            page.wait_for_timeout(500)
            return True

        try:
            body_text = body.inner_text()
        except Exception:
            body_text = ""

        signature = (len(body_text), body_text[:500])

        if signature == last_signature:
            stable_checks += 1
        else:
            stable_checks = 0
            last_signature = signature

        if stable_checks >= 6:
            print(f"Skipping ticket {ticket_number}: no visible image cards detected after detail page stabilized.")
            return False


def extract_full_image_url(card):
    gross_anchor = card.locator('a:has(img[title*="Gross Image"])').first
    if gross_anchor.count() > 0:
        href = gross_anchor.get_attribute("href")
        if href and "GetActualSizeTicketItemPicture" in href:
            return urljoin(BASE_URL, href), "Gross Image (anchor)"

    direct_img = card.locator('img[id^="img_"][src*="GetActualSizeTicketItemPicture"]').first
    if direct_img.count() > 0:
        src = direct_img.get_attribute("src")
        if src:
            return urljoin(BASE_URL, src), "Direct Image Src"

    gross_img = card.locator('img[title*="Gross Image"]').first
    if gross_img.count() > 0:
        src = gross_img.get_attribute("src")
        if src and "GetActualSizeTicketItemPicture" in src:
            return urljoin(BASE_URL, src), "Gross Image Src"

    return None, None


def prepare_ticket_image_tasks(page, ticket_number: str):
    print(f"Downloading images for {ticket_number}...")

    has_images = wait_for_ticket_detail_or_no_images(page, ticket_number)
    if not has_images:
        return []

    item_cards = page.locator("div.ticketItem:visible")
    card_count = item_cards.count()

    print(f"Found {card_count} ticket item card(s).")

    tasks = []

    for i in range(card_count):
        card = item_cards.nth(i)

        try:
            label = card.locator("div.ticketItemMaterial").inner_text().strip()
        except Exception:
            print(f"Could not read material label for {ticket_number} item {i+1}.")
            continue

        safe_label = clean_label(label)

        full_url, chosen_type = extract_full_image_url(card)
        if not full_url:
            print(f"No direct full-size image URL found for {ticket_number} item {i+1} ({label}).")
            continue

        try:
            response = page.context.request.get(full_url)
            if not response.ok:
                print(f"Download request failed for {ticket_number} item {i+1}: {response.status}")
                continue

            content_type = response.headers.get("content-type", "")
            ext = extension_from_content_type(content_type)
            filename = f"{ticket_number}_{i+1:02d}_{safe_label}{ext}"
            image_bytes = response.body()

            dataset_split = choose_dataset_split(
                ticket_number=ticket_number,
                item_index=i + 1,
                material_label=label,
                filename=filename,
            )

            tasks.append({
                "ticket_number": ticket_number,
                "item_index": i + 1,
                "material_label": label,
                "safe_label": safe_label,
                "saved_filename": filename,
                "dataset_split": dataset_split,
                "full_image_url": full_url,
                "image_choice": chosen_type,
                "content_type": content_type,
                "image_bytes": image_bytes,
            })

        except Exception as e:
            print(f"Failed to download image for {ticket_number} item {i+1}: {e}")

    if not tasks:
        print(f"Skipping ticket {ticket_number}: no usable image tasks found.")

    return tasks


def upload_single_task(task):
    dataset_s3_uri, ticket_lookup_s3_uri = upload_bytes_to_s3(
        image_bytes=task["image_bytes"],
        dataset_split=task["dataset_split"],
        material_label=task["material_label"],
        ticket_number=task["ticket_number"],
        filename=task["saved_filename"],
        content_type=task["content_type"],
    )

    return dataset_s3_uri, ticket_lookup_s3_uri, task["safe_label"], task["saved_filename"], task["dataset_split"], task["ticket_number"], task["image_choice"]


def download_ticket_images(page, ticket_number: str):
    ticket_tasks = prepare_ticket_image_tasks(page, ticket_number)

    if not ticket_tasks:
        return []

    saved_files = []

    with ThreadPoolExecutor(max_workers=min(S3_UPLOAD_WORKERS, len(ticket_tasks))) as executor:
        futures = [executor.submit(upload_single_task, task) for task in ticket_tasks]

        for future in as_completed(futures):
            try:
                dataset_s3_uri, ticket_lookup_s3_uri, safe_label, saved_filename, dataset_split, ticket_number, image_choice = future.result()
                saved_files.append(dataset_s3_uri)

                print(
                    f"Uploaded {saved_filename} to "
                    f"{dataset_split}/{safe_label}/ and "
                    f"{ALL_TICKETS_DIR}/{ticket_number}/ using "
                    f"{image_choice}"
                )
            except Exception as e:
                print(f"Failed to upload task: {e}")

    return saved_files


def maybe_reset_search_session(page, current_page: int, start_date: str, end_date: str):
    if current_page > 1 and (current_page - 1) % RESET_EVERY_N_PAGES == 0:
        print(f"\nResetting page/session before processing page {current_page}...")
        jump_to_page(page, current_page, start_date, end_date)


def process_ticket_results(page, start_date: str, end_date: str):
    processed_tickets = 0
    current_page = 1

    while True:
        if current_page > MAX_PAGES_TO_PROCESS:
            print(f"Reached page limit of {MAX_PAGES_TO_PROCESS}. Stopping.")
            break

        maybe_reset_search_session(page, current_page, start_date, end_date)

        print(f"\nProcessing results page {current_page}...")
        ticket_numbers = read_current_page_ticket_numbers(page)

        for ticket_number in ticket_numbers:
            if processed_tickets >= MAX_TICKETS_TO_PROCESS:
                print(f"Reached ticket limit of {MAX_TICKETS_TO_PROCESS}. Stopping.")
                return

            opened = click_ticket_row(page, ticket_number)
            if not opened:
                continue

            try:
                saved_files = download_ticket_images(page, ticket_number)
                print(f"Uploaded {len(saved_files)} image(s) for {ticket_number}")
            except Exception as e:
                print(f"Error while processing {ticket_number}: {e}")

            processed_tickets += 1
            page.wait_for_timeout(800)

        if processed_tickets >= MAX_TICKETS_TO_PROCESS:
            print(f"Reached ticket limit of {MAX_TICKETS_TO_PROCESS}. Stopping.")
            break

        moved = click_next_page(page)
        if not moved:
            break

        current_page += 1
        page.wait_for_timeout(1200)


def main():
    args = parse_args()
    start_date, end_date = resolve_date_range(args)

    print(f"Resolved date range: {start_date} -> {end_date}")
    print(f"Valid ticket prefixes: {sorted(VALID_TICKET_PREFIXES)}")

    # The whole browser lifecycle is inside try/except now. Browser launch,
    # context creation and new_page() used to sit outside the try, so a
    # Chromium launch failure escaped uncaught -> Python exited non-zero.
    # Any failure here exits non-zero so the run is diagnosable; the
    # instance stop is handled unconditionally by weekly_scrape.sh's EXIT trap.
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                context.set_default_timeout(DEFAULT_TIMEOUT_MS)
                context.set_default_navigation_timeout(NAV_TIMEOUT_MS)
                page = context.new_page()

                login(page)
                wait_for_login_to_complete(page)
                run_search(page, start_date, end_date)

                process_ticket_results(page, start_date, end_date)

                print("\nFinished scraping and uploading images.")
                print(f"S3 bucket used: {S3_BUCKET_NAME}")
                print(
                    f"Dataset structure used: train/<label>/, val/<label>/, "
                    f"test/<label>/, {ALL_TICKETS_DIR}/<ticket_number>/"
                )
            finally:
                browser.close()

    except PlaywrightTimeoutError as e:
        print(f"\nPlaywright timeout: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
