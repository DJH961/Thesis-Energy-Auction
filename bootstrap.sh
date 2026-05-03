#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — single-file launcher for Azure ML
# =============================================================================
# This file is designed so you NEVER need to edit it. Upload it to Azure ML
# (optionally together with a custom sweep YAML) and run:
#
#     bash bootstrap.sh
#
# What it does:
#   1. Pulls the latest source from the GitHub repo's `main` branch (so the
#      "version" you run is whatever is currently on `main`; pinning to a
#      tag/commit is opt-in via the REF env var, no edit needed).
#   2. Picks a sweep spec to run, in this order:
#        (i)   $SWEEP_SPEC env var, if set;
#        (ii)  any *.yaml/*.yml file uploaded alongside this script (so you
#              can upload bootstrap.sh + my_new_sweep.yaml together without
#              touching the repo);
#        (iii) configs/sweeps/default_vs_unsold_to_msr.yaml (repo default).
#   3. Auto-installs `pip install -r requirements.txt` only when one of the
#      critical Python modules (numpy, pandas, pyarrow, yaml, torch,
#      gymnasium) is missing from the worker's environment, so a curated
#      Azure ML env that lacks pyarrow doesn't silently strand the
#      end-of-run CSV→parquet compression. Set INSTALL_DEPS=1 to force a
#      fresh install.
#   4. Copies the produced results/ tree into the Azure ML output directory.
#
# Optional env vars (set in the Azure ML "environment variables" panel; never
# requires editing this file):
#     REF            git ref to fetch (default: main).      e.g. v8.5.4
#     SWEEP_SPEC     sweep spec path inside the repo, OR
#                    the basename of an uploaded yaml.       e.g. my_sweep.yaml
#     INSTALL_DEPS   set to 1 to force `pip install -r requirements.txt`.
#     REPO_SLUG      override repo (default: DJH961/Thesis-Energy-Auction).
#
# Optional CLI arg:
#     $1 = output directory (typically `${{outputs.results}}`); falls back to
#     ./outputs which Azure ML auto-uploads.
# =============================================================================

set -euo pipefail

# ----- Configuration ---------------------------------------------------------
REF="${REF:-main}"
REPO_SLUG="${REPO_SLUG:-DJH961/Thesis-Energy-Auction}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
OUTPUT_DIR="${1:-./outputs}"

# Capture the directory the user uploaded this script in BEFORE we extract
# the repo on top of it (so we can find any sibling yaml uploads).
UPLOAD_DIR="$(pwd)"

echo "================================================================="
echo "  ETS MARL bootstrap"
echo "  ref         : ${REF}"
echo "  repo        : ${REPO_SLUG}"
echo "  install_deps: ${INSTALL_DEPS}"
echo "  output dir  : ${OUTPUT_DIR}"
echo "  upload dir  : ${UPLOAD_DIR}"
echo "================================================================="

mkdir -p "${OUTPUT_DIR}"

# ----- 0. Stash any uploaded sweep yaml(s) before the repo overwrites them ---
# We look for yaml files that are NOT this script's directory's well-known
# bootstrap artefacts. Anything user-uploaded that ends in .yaml/.yml counts.
STASH_DIR="$(mktemp -d)"
shopt -s nullglob
for f in "${UPLOAD_DIR}"/*.yaml "${UPLOAD_DIR}"/*.yml; do
    cp -- "$f" "${STASH_DIR}/"
    echo "[bootstrap] Stashed uploaded yaml: $(basename "$f")"
done
shopt -u nullglob

# ----- 1. Fetch source --------------------------------------------------------
# Use the codeload tarball endpoint, which works for both branch names and tags.
TARBALL_URL="https://codeload.github.com/${REPO_SLUG}/tar.gz/refs/heads/${REF}"
# If REF doesn't look like a branch (e.g. starts with 'v' for a tag, or is a
# 40-char SHA), try the generic tarball endpoint which resolves both.
case "${REF}" in
    v*|V*) TARBALL_URL="https://codeload.github.com/${REPO_SLUG}/tar.gz/refs/tags/${REF}" ;;
esac
echo "[bootstrap] Downloading ${TARBALL_URL}"
if ! curl -fsSL "${TARBALL_URL}" | tar -xz --strip-components=1; then
    # Fallback: generic ref endpoint handles tags, branches, and SHAs.
    echo "[bootstrap] First URL failed; falling back to generic tarball endpoint"
    curl -fsSL "https://github.com/${REPO_SLUG}/archive/${REF}.tar.gz" \
        | tar -xz --strip-components=1
fi

# Print the version actually checked out, for the run log.
if [ -f pyproject.toml ]; then
    VERSION_LINE="$(grep -E '^version' pyproject.toml || true)"
    echo "[bootstrap] Repo version: ${VERSION_LINE}"
fi

# ----- 2. Select sweep spec ---------------------------------------------------
# Priority: SWEEP_SPEC env > uploaded yaml > repo default.
DEFAULT_SPEC="configs/sweeps/default_vs_unsold_to_msr.yaml"
SPEC_TO_RUN=""

if [ -n "${SWEEP_SPEC:-}" ]; then
    # Allow SWEEP_SPEC to name either a path inside the repo or a basename
    # of an uploaded yaml.
    if [ -f "${SWEEP_SPEC}" ]; then
        SPEC_TO_RUN="${SWEEP_SPEC}"
    elif [ -f "${STASH_DIR}/${SWEEP_SPEC}" ]; then
        SPEC_TO_RUN="${STASH_DIR}/${SWEEP_SPEC}"
    else
        echo "[bootstrap] ERROR: SWEEP_SPEC='${SWEEP_SPEC}' not found in repo or uploads"
        exit 2
    fi
else
    # Pick the first uploaded yaml, if any.
    shopt -s nullglob
    UPLOADED_YAMLS=( "${STASH_DIR}"/*.yaml "${STASH_DIR}"/*.yml )
    shopt -u nullglob
    if [ "${#UPLOADED_YAMLS[@]}" -gt 0 ]; then
        SPEC_TO_RUN="${UPLOADED_YAMLS[0]}"
        if [ "${#UPLOADED_YAMLS[@]}" -gt 1 ]; then
            echo "[bootstrap] WARNING: multiple uploaded yamls; using $(basename "${SPEC_TO_RUN}")."
            echo "[bootstrap]          Set SWEEP_SPEC=<basename> to disambiguate."
        fi
    else
        SPEC_TO_RUN="${DEFAULT_SPEC}"
    fi
fi
echo "[bootstrap] Using sweep spec: ${SPEC_TO_RUN}"

# ----- 3. Optional dependency install ----------------------------------------
# Self-heal: even with INSTALL_DEPS=0, a curated Azure ML environment may be
# missing one or more of the heavy deps the trainer + lossless end-of-run
# compression need (pyarrow in particular is the silent failure that strands
# multi-GB CSVs / checkpoint dirs uncompressed). Probe for the critical
# imports and install requirements.txt if any are absent.
_NEEDED_PY_MODS=(numpy pandas pyarrow yaml torch gymnasium)
# Keep this list aligned with requirements.txt / pyproject.toml. It only
# needs to cover modules whose absence would silently degrade a run
# (e.g. pyarrow for the end-of-run lossless log compression). Other deps
# get pulled in transitively when pip install -r requirements.txt runs.
_MISSING_MODS=""
for _mod in "${_NEEDED_PY_MODS[@]}"; do
    if ! python -c "import ${_mod}" >/dev/null 2>&1; then
        _MISSING_MODS="${_MISSING_MODS} ${_mod}"
    fi
done

if [ "${INSTALL_DEPS}" = "1" ]; then
    echo "[bootstrap] INSTALL_DEPS=1 -> installing requirements.txt"
    python -m pip install --upgrade pip -q
    python -m pip install -r requirements.txt -q
elif [ -n "${_MISSING_MODS}" ]; then
    echo "[bootstrap] Missing Python modules detected:${_MISSING_MODS}"
    echo "[bootstrap] Auto-installing requirements.txt so end-of-run lossless"
    echo "[bootstrap] CSV->parquet / checkpoint->tar.xz compression can run."
    python -m pip install --upgrade pip -q
    python -m pip install -r requirements.txt -q
else
    echo "[bootstrap] Skipping pip install (INSTALL_DEPS!=1 and key modules already present)."
fi

# ----- 4. Run the sweep -------------------------------------------------------
echo "[bootstrap] Running: python scripts/sweep.py --spec ${SPEC_TO_RUN}"
SWEEP_RC=0
python scripts/sweep.py --spec "${SPEC_TO_RUN}" || SWEEP_RC=$?
echo "[bootstrap] Sweep exit code: ${SWEEP_RC}"

# ----- 5. Publish results -----------------------------------------------------
if [ -d "results" ]; then
    echo "[bootstrap] Copying results/ -> ${OUTPUT_DIR}/"
    cp -r results/. "${OUTPUT_DIR}/"
else
    echo "[bootstrap] WARNING: no results/ directory was produced"
fi

echo "[bootstrap] Done. Output listing:"
ls -la "${OUTPUT_DIR}" || true

exit "${SWEEP_RC}"
