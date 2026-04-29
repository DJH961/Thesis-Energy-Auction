#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — single-file launcher for Azure ML
# =============================================================================
# This is the ONLY file you upload as the "code" source in the Azure ML job
# wizard. It downloads a tagged release of the repository, installs Python
# dependencies, runs a sweep, and copies the results to the Azure ML output
# directory.
#
# Usage in the Azure ML "command" box:
#     bash bootstrap.sh ${{outputs.results}}
#
# Or, if you have not configured a named output and just want Azure ML to
# auto-upload everything written to ./outputs (the default behaviour):
#     bash bootstrap.sh
#
# -----------------------------------------------------------------------------
# Next experiment workflow:
#   1. Locally:  git tag vX.Y.Z-my-experiment && git push origin vX.Y.Z-my-experiment
#   2. Edit the TAG variable below (or pass TAG=... as an env var in the
#      Azure ML environment-variables section — no file edit required).
#   3. Optionally edit SWEEP_SPEC to point at a different sweep config.
#   4. Re-upload this single file (or just re-run the job if you used env vars).
# =============================================================================

set -euo pipefail

# ----- Configuration (only thing you normally need to change) ----------------
TAG="${TAG:-v8.5.4}"
SWEEP_SPEC="${SWEEP_SPEC:-configs/sweeps/default_vs_unsold_to_msr.yaml}"
REPO_SLUG="${REPO_SLUG:-DJH961/Thesis-Energy-Auction}"

# Output destination: first CLI arg (typically ${{outputs.results}} from
# Azure ML), or ./outputs which Azure ML auto-uploads by default.
OUTPUT_DIR="${1:-./outputs}"

echo "================================================================="
echo "  ETS MARL bootstrap"
echo "  tag        : ${TAG}"
echo "  sweep spec : ${SWEEP_SPEC}"
echo "  output dir : ${OUTPUT_DIR}"
echo "  workdir    : $(pwd)"
echo "================================================================="

mkdir -p "${OUTPUT_DIR}"

# ----- 1. Fetch tagged source -------------------------------------------------
TARBALL_URL="https://github.com/${REPO_SLUG}/archive/refs/tags/${TAG}.tar.gz"
echo "[bootstrap] Downloading ${TARBALL_URL}"
curl -fsSL "${TARBALL_URL}" | tar -xz --strip-components=1

# ----- 2. Install Python dependencies ----------------------------------------
echo "[bootstrap] Installing requirements"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

# ----- 3. Run the sweep -------------------------------------------------------
echo "[bootstrap] Running sweep: ${SWEEP_SPEC}"
# Do not let a non-zero exit from the sweep skip the result-copy step; we still
# want partial outputs uploaded for inspection.
SWEEP_RC=0
python scripts/sweep.py --spec "${SWEEP_SPEC}" || SWEEP_RC=$?
echo "[bootstrap] Sweep exit code: ${SWEEP_RC}"

# ----- 4. Publish results to the Azure ML output directory --------------------
# The sweep spec's `output_dir` lives under results/sweeps/<name>/; copy
# whatever ended up under results/ into the Azure ML output mount.
if [ -d "results" ]; then
    echo "[bootstrap] Copying results/ -> ${OUTPUT_DIR}/"
    # -T-style copy of contents (works with both GNU and BusyBox cp).
    cp -r results/. "${OUTPUT_DIR}/"
else
    echo "[bootstrap] WARNING: no results/ directory was produced"
fi

echo "[bootstrap] Done. Final output listing:"
ls -la "${OUTPUT_DIR}" || true

exit "${SWEEP_RC}"
