#!/usr/bin/env bash
set -euo pipefail

# Windows path to the directory above the run folder.
WINDOWS_PROJECT_DIR='D:\PISO-runs'
OUTPUT_RUN='classification'

# Add or remove exact method names here.
METHODS_TO_REMOVE=(
  # "GZO_NS"
  # "GZO_HS"
  # "ZO_TG"
  "ZO_OG"
  # "ZO_OGVR"
  # "GaussianPISO"
  # "CyclePISO"
  # "GaussianPISO2"
  # "CyclePISO2"
)

KNOWN_METHODS=(
  "GaussianPISO2"
  "CyclePISO2"
  "GaussianPISO"
  "CyclePISO"
  "ZO_OGVR"
  "GZO_NS"
  "GZO_HS"
  "ZO_TG"
  "ZO_OG"
)

if command -v cygpath >/dev/null 2>&1; then
  PROJECT_DIR="$(cygpath -u "$WINDOWS_PROJECT_DIR")"
elif command -v wslpath >/dev/null 2>&1; then
  PROJECT_DIR="$(wslpath -u "$WINDOWS_PROJECT_DIR")"
else
  echo "Could not convert the Windows path."
  exit 1
fi

CACHE_DIR="${PROJECT_DIR%/}/${OUTPUT_RUN}/cache"
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "Cache directory not found: $CACHE_DIR"
  exit 1
fi

is_selected() {
  local method="$1"
  local selected
  for selected in "${METHODS_TO_REMOVE[@]}"; do
    [[ "$method" == "$selected" ]] && return 0
  done
  return 1
}

detect_method() {
  local filename="$1"
  local method
  for method in "${KNOWN_METHODS[@]}"; do
    if [[ "$filename" == "${method}_"* ]]; then
      printf '%s\n' "$method"
      return 0
    fi
  done
  return 1
}

declare -a FILES_TO_DELETE=()
while IFS= read -r -d '' file; do
  filename="$(basename "$file")"
  if method="$(detect_method "$filename")" && is_selected "$method"; then
    FILES_TO_DELETE+=("$file")
  fi
done < <(find "$CACHE_DIR" -type f \( -name '*.csv' -o -name '*.tmp.csv' \) -print0)

if (( ${#FILES_TO_DELETE[@]} == 0 )); then
  echo "No matching cache files found in $CACHE_DIR"
  exit 0
fi

printf 'Deleting %d cache file(s):\n' "${#FILES_TO_DELETE[@]}"
printf '  %s\n' "${FILES_TO_DELETE[@]}"
rm -f -- "${FILES_TO_DELETE[@]}"
rm -f -- "$CACHE_DIR/cache_manifest.csv"

echo "Done. Aggregate outputs will be regenerated on the next run."
