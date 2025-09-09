set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "${RAW_DIR}"

MODE="${1:-csv}" 
RETRIES=5
SLEEP=3

download_with_check() {
  local url="$1"
  local out="$2"
  local tmp="${out}.part"

  echo "[INFO] Downloading: ${url}"
  rm -f "${tmp}"
  for i in $(seq 1 ${RETRIES}); do
    echo "[INFO] Attempt ${i}/${RETRIES}"
    if curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 15 --output "${tmp}" "${url}"; then
      if [[ "${out}" == *.gz ]]; then
        if gzip -t "${tmp}"; then
          echo "[INFO] Gzip OK"
          mv -f "${tmp}" "${out}"
          return 0
        else
          echo "[WARN] Gzip check failed, retrying after ${SLEEP}s..."
          sleep "${SLEEP}"
          continue
        fi
      else
        mv -f "${tmp}" "${out}"
        return 0
      fi
    else
      echo "[WARN] curl failed, retrying after ${SLEEP}s..."
      sleep "${SLEEP}"
    fi
  done

  echo "[FATAL] Failed to download valid file after ${RETRIES} attempts: ${url}"
  rm -f "${tmp}"
  exit 1
}

if [[ "${MODE}" == "csv" ]]; then
  URL="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"  
  OUT="${RAW_DIR}/en.openfoodfacts.org.products.csv.gz"
  download_with_check "${URL}" "${OUT}"
  echo "[INFO] Saved: ${OUT}"
  echo "[INFO] Lines (gzip):"
  gzip -cd "${OUT}" | wc -l || true

elif [[ "${MODE}" == "jsonl" ]]; then
  URL="https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"
  OUT="${RAW_DIR}/openfoodfacts-products.jsonl.gz"
  download_with_check "${URL}" "${OUT}"
  echo "[INFO] Saved: ${OUT}"
  echo "[INFO] Lines (gzip):"

  gzip -cd "${OUT}" | wc -l || true

else
  echo "[FATAL] Unknown mode: ${MODE}. Use 'csv' or 'jsonl'."
  exit 1
fi

