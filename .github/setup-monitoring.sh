#!/usr/bin/env bash
# Uptime monitoring for the UnBias Plus vLLM service.
#
# Run once, as a project admin on unbias-toolkit:
#   bash .github/setup-monitoring.sh you@vectorinstitute.ai
#
# Why this exists: on 2026-07-29 the vLLM service began failing every cold
# start (CUDA driver mismatch) and nothing reported it — the CI deploy gate
# only runs at deploy time, and this break came from a Cloud Run host driver
# upgrade, not from a deploy. An uptime check catches a running service that
# has gone bad, which is the gap CI cannot cover.
#
# Safe to re-run: every resource is created only if absent.

set -euo pipefail

PROJECT_ID="unbias-toolkit"
REGION="us-east4"
SERVICE="unbias-plus-vllm"
EMAIL="${1:-}"

if [ -z "$EMAIL" ]; then
  echo "usage: bash .github/setup-monitoring.sh <alert-email>" >&2
  exit 1
fi

gcloud config set project "${PROJECT_ID}" >/dev/null

HOST=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" --project="${PROJECT_ID}" \
  --format='value(status.url)' | sed 's#^https://##')

if [ -z "$HOST" ]; then
  echo "✗ could not resolve the URL for ${SERVICE}" >&2
  exit 1
fi
echo "=== Monitoring ${HOST} ==="

# ── 1. Notification channel ───────────────────────────────────────────────────
CHANNEL=$(gcloud alpha monitoring channels list \
  --project="${PROJECT_ID}" \
  --filter="labels.email_address='${EMAIL}'" \
  --format='value(name)' | head -1)

if [ -z "$CHANNEL" ]; then
  CHANNEL=$(gcloud alpha monitoring channels create \
    --project="${PROJECT_ID}" \
    --display-name="UnBias Plus alerts" \
    --type=email \
    --channel-labels="email_address=${EMAIL}" \
    --format='value(name)')
  echo "✓ created notification channel for ${EMAIL}"
  echo "  NOTE: confirm the verification email before alerts will deliver."
else
  echo "✓ notification channel for ${EMAIL} already exists"
fi

# ── 2. Uptime check on /health ────────────────────────────────────────────────
# Every 5 minutes from multiple regions. A crash-looping revision serves 500,
# so this fires whether the container is down or merely unhealthy.
if gcloud monitoring uptime list-configs --project="${PROJECT_ID}" \
     --filter="displayName='${SERVICE} health'" --format='value(name)' | grep -q .; then
  echo "✓ uptime check already exists"
else
  gcloud monitoring uptime create "${SERVICE} health" \
    --project="${PROJECT_ID}" \
    --resource-type=uptime-url \
    --resource-labels="host=${HOST},project_id=${PROJECT_ID}" \
    --path="/health" \
    --port=443 \
    --protocol=https \
    --period=5 \
    --timeout=30 >/dev/null
  echo "✓ created uptime check on https://${HOST}/health"
fi

CHECK_ID=$(gcloud monitoring uptime list-configs --project="${PROJECT_ID}" \
  --filter="displayName='${SERVICE} health'" --format='value(name)' | head -1 | sed 's#.*/##')

# ── 3. Alert policy ───────────────────────────────────────────────────────────
# Fires when the check has been failing for 10 minutes, so a single blip or an
# ordinary cold start does not page anyone.
POLICY_NAME="${SERVICE} is down"
if gcloud alpha monitoring policies list --project="${PROJECT_ID}" \
     --filter="displayName='${POLICY_NAME}'" --format='value(name)' | grep -q .; then
  echo "✓ alert policy already exists"
else
  TMP=$(mktemp)
  cat >"${TMP}" <<JSON
{
  "displayName": "${POLICY_NAME}",
  "combiner": "OR",
  "conditions": [{
    "displayName": "/health failing for 10 minutes",
    "conditionThreshold": {
      "filter": "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\" AND metric.label.check_id=\"${CHECK_ID}\"",
      "aggregations": [{
        "alignmentPeriod": "300s",
        "perSeriesAligner": "ALIGN_FRACTION_TRUE"
      }],
      "comparison": "COMPARISON_LT",
      "thresholdValue": 1,
      "duration": "600s",
      "trigger": { "count": 1 }
    }
  }],
  "notificationChannels": ["${CHANNEL}"],
  "documentation": {
    "content": "unbias-plus-vllm is not serving. Check for a crash-looping revision:\n\n  gcloud run revisions list --service=${SERVICE} --region=${REGION} --project=${PROJECT_ID}\n  gcloud logging read 'resource.labels.service_name=\"${SERVICE}\" AND severity>=ERROR' --project=${PROJECT_ID} --limit=50\n\nA CUDA error 803 here means the Cloud Run host driver moved and the image needs rebuilding.",
    "mimeType": "text/markdown"
  }
}
JSON
  gcloud alpha monitoring policies create --project="${PROJECT_ID}" --policy-from-file="${TMP}" >/dev/null
  rm -f "${TMP}"
  echo "✓ created alert policy '${POLICY_NAME}' → ${EMAIL}"
fi

echo ""
echo "Done. Verify at: https://console.cloud.google.com/monitoring/uptime?project=${PROJECT_ID}"
