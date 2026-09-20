#!/bin/bash
set -euo pipefail

export HOME="/home/ubuntu"
export PLAYWRIGHT_BROWSERS_PATH="/home/ubuntu/.cache/ms-playwright"
export PATH="/home/ubuntu/MVMR/venv/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Guaranteed self-stop.
#
# The stop lives in a trap on EXIT so the instance is ALWAYS shut down no
# matter how the scraper terminates: clean exit, Python crash, OOM kill,
# non-zero exit under `set -e`, or an unexpected signal. Previously the stop
# was the last line of the script, so any non-zero exit from Python aborted
# the script (set -e) before it ran and the box stayed up billing forever.
# ---------------------------------------------------------------------------
stop_instance() {
  # Preserve the scraper's exit code so it is visible in the logs.
  local exit_code=$?
  echo "Scraper finished with exit code ${exit_code}. Stopping instance..."

  # `set -e` is disabled inside the trap so a transient metadata/AWS hiccup
  # can't prevent the stop attempt from completing.
  set +e

  local token instance_id region
  token=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)

  instance_id=$(curl -H "X-aws-ec2-metadata-token: ${token}" -s \
    http://169.254.169.254/latest/meta-data/instance-id)

  region=$(curl -H "X-aws-ec2-metadata-token: ${token}" -s \
    http://169.254.169.254/latest/dynamic/instance-identity/document | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['region'])")

  echo "Stopping ${instance_id} in ${region}..."
  /usr/local/bin/aws ec2 stop-instances \
    --instance-ids "${instance_id}" --region "${region}"
}
trap stop_instance EXIT

cd /home/ubuntu/MVMR
source /home/ubuntu/MVMR/venv/bin/activate

echo "Starting weekly scrape at $(date --iso-8601=seconds)..."
python -u /home/ubuntu/MVMR/scrape_tickets.py --mode weekly
echo "Scraper returned cleanly."
