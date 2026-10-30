#!/usr/bin/env bash
#
# build_report.sh — compile the thesis report to PDF.
#
# Prefers tectonic (self-contained), falls back to latexmk, then to a manual
# pdflatex + bibtex cycle. Leaves `report/main.pdf` fresh and, when an output
# path is given as an argument (or via $UNDERTOW_REPORT_PDF), copies the PDF
# there so a viewable rendering is always available outside the repo. A
# relative output path resolves against the caller's working directory, not the
# report directory.
#
# Run from the repository root:  bash scripts/build_report.sh [output.pdf]
# Env:    UNDERTOW_REPORT_PDF  default copy destination when no argument is given
#         TECTONIC             explicit path to the tectonic binary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${REPO_ROOT}/report"
CALLER_PWD="$(pwd)"

if [[ ! -f "${REPORT_DIR}/main.tex" ]]; then
  echo "error: ${REPORT_DIR}/main.tex not found" >&2
  exit 2
fi

cd "${REPORT_DIR}"

engine=""
if [[ -n "${TECTONIC:-}" ]]; then
  if [[ -x "${TECTONIC}" ]]; then
    engine="${TECTONIC}"
  else
    echo "warning: TECTONIC=${TECTONIC} is not executable; falling back" >&2
  fi
fi
if [[ -z "${engine}" ]] && command -v tectonic >/dev/null 2>&1; then
  engine="tectonic"
fi
if [[ -z "${engine}" && -x "${HOME:-}/.local/bin/tectonic" ]]; then
  engine="${HOME:-}/.local/bin/tectonic"
fi

if [[ -n "${engine}" ]]; then
  "${engine}" main.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
else
  echo "error: no LaTeX engine found (tectonic, latexmk or pdflatex)" >&2
  exit 2
fi

pdf="${REPORT_DIR}/main.pdf"
if [[ ! -f "${pdf}" ]]; then
  echo "error: the build produced no ${pdf}" >&2
  exit 2
fi
echo "built ${pdf}"

destination="${1:-${UNDERTOW_REPORT_PDF:-}}"
if [[ -n "${destination}" && "${destination}" != /* ]]; then
  destination="${CALLER_PWD}/${destination}"
fi
if [[ -n "${destination}" ]]; then
  mkdir -p "$(dirname "${destination}")"
  cp "${pdf}" "${destination}"
  echo "copied ${destination}"
fi
