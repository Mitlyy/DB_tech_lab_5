set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "${RAW_DIR}"

URL="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
OUT="${RAW_DIR}/en.openfoodfacts.org.products.csv.gz"

echo "[INFO] Downloading: ${URL}"
curl -L --fail --retry 3 --output "${OUT}" "${URL}"

echo "[INFO] File saved to: ${OUT}"
echo "[INFO] Gz lines count (quick sanity check):"
gzip -cd "${OUT}" | wc -l || true

