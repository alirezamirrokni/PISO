#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash clear_method_caches.sh routing GaussianPISO
#   bash clear_method_caches.sh security PISO
#   bash clear_method_caches.sh routing PISO2
#   bash clear_method_caches.sh routing all

FAMILY="${1:-}"
shift || true
if [[ "$FAMILY" != "routing" && "$FAMILY" != "security" ]]; then
  echo "Usage: $0 routing|security METHOD [METHOD ...]"
  exit 2
fi

if [[ "$FAMILY" == "routing" ]]; then
  WINDOWS_OUTPUT='D:/piso-runs/routing game'
else
  WINDOWS_OUTPUT='D:/piso-runs/security game'
fi

convert_path() {
  local path="$1"
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$path"
  elif command -v wslpath >/dev/null 2>&1; then
    wslpath -u "$path"
  else
    printf '%s\n' "$path"
  fi
}

OUTPUT="$(convert_path "$WINDOWS_OUTPUT")"
CACHE="$OUTPUT/cache"
REQUESTED=("$@")
if (( ${#REQUESTED[@]} == 0 )); then
  echo "Specify at least one method, ZO, GZO, PISO, PISO2, or all."
  exit 2
fi

ALL_METHODS=(
  ZOS ZO_TG ZO_OG ZO_OGVR GZO_NS GZO_HS PZOS
  GaussianPISO CyclePISO
  GaussianPISO2 CyclePISO2
)
ZO_METHODS=(ZO_TG ZO_OG ZO_OGVR)
GZO_METHODS=(GZO_NS GZO_HS)
PISO_METHODS=(GaussianPISO CyclePISO)
PISO2_METHODS=(GaussianPISO2 CyclePISO2)

expand_request() {
  local item="$1"
  case "$item" in
    ZO) printf '%s\n' "${ZO_METHODS[@]}" ;;
    GZO) printf '%s\n' "${GZO_METHODS[@]}" ;;
    PISO) printf '%s\n' "${PISO_METHODS[@]}" ;;
    PISO2) printf '%s\n' "${PISO2_METHODS[@]}" ;;
    all) printf '%s\n' "${ALL_METHODS[@]}" ;;
    ZOS|ZO_TG|ZO_OG|ZO_OGVR|GZO_NS|GZO_HS|PZOS|GaussianPISO|CyclePISO|GaussianPISO2|CyclePISO2)
      printf '%s\n' "$item" ;;
    *) echo "Unknown method or group: $item" >&2; exit 2 ;;
  esac
}

mapfile -t METHODS < <(
  for item in "${REQUESTED[@]}"; do expand_request "$item"; done | sort -u
)

for method in "${METHODS[@]}"; do
  echo "Removing: $CACHE/$method"
  rm -rf -- "$CACHE/$method"
done

rm -f -- \
  "$OUTPUT/raw_runs.csv" \
  "$OUTPUT/normalized_runs.csv" \
  "$OUTPUT/trajectory_summary.csv" \
  "$OUTPUT/final_scores.csv" \
  "$OUTPUT/config.yaml"
rm -f -- "$OUTPUT"/routing_*.png "$OUTPUT"/routing_*.pdf "$OUTPUT"/routing_*.csv
rm -f -- "$OUTPUT"/security_*.png "$OUTPUT"/security_*.pdf "$OUTPUT"/security_*.csv

echo "Done. Problem instances and all unselected method caches were preserved."
