# Undertow thesis report

The submission-form research report for the Undertow project: a two-column
IEEE paper written in LaTeX. Every rule about how the report is written lives
in [`docs/agents/report-writer.md`](../docs/agents/report-writer.md) — read it
before editing prose.

## Build

A LaTeX toolchain is required, but the wrapper picks whichever engine exists.
Run it from the repository root:

```bash
bash scripts/build_report.sh                          # tectonic -> latexmk -> pdflatex
UNDERTOW_REPORT_PDF=~/report-build/thesis.pdf bash scripts/build_report.sh
```

The compiled `main.pdf` and the aux files are gitignored: only the sources are
versioned. Every PR that touches `report/` also attaches the compiled PDF to
its GitHub Actions run under the `thesis-report-pdf` artifact, so the current
rendering is available without a local toolchain.

## Structure

```
main.tex                  IEEEtran setup, title, abstract, input list, \pending{} macro
sections/00_abstract.tex
sections/01_introduction.tex ... 09_conclusion.tex
references.bib            verified BibTeX entries (IEEE numeric style)
```

## Plagiarism screen

The free local checker compares the report prose against the cited open-access
PDFs and the repo's own docs at the 8-consecutive-word threshold:

```bash
uv run python scripts/plagiarism_check.py                # downloads missing PDFs into data/plagiarism_corpus/
uv run python scripts/plagiarism_check.py --no-download  # cached corpus only
```

External matches (the cited papers) fail the check with exit code 1 and must be
rewritten. Internal overlaps (README, ADRs, plans) are warnings; `--strict-internal`
turns them into failures too. Exit code 2 means the check itself is incomplete
or misconfigured: an unreadable source, an empty report directory, or a bad
manifest. `--groups external|internal` restricts the scan; every group in the
scan must have at least one resolvable document in the manifest. The corpus and
its URLs live in `configs/plagiarism_corpus.toml`. Turnitin at submission
remains the final gate.

## Rules of the house (summary)

- **Original prose only.** Never copy text from a paper, abstract, whitepaper or
  from the repo's own docs. Eight or more consecutive words matching a source
  means rewrite the sentence.
- **IEEE references.** Every borrowed claim is cited with `\cite{...}` through
  `IEEEtran.bst`; every `references.bib` entry is verified against its primary
  source before it is added; no fabricated venues.
- **No invented numbers.** A number that does not yet exist is written as
  `\pending{what is missing}`. Before submission, every `\pending{}` and
  `\todo{}` must be resolved; `grep -n '\\pending\|\\todo' sections/*.tex`
  lists them.
- **Evidence.** Results come from locked runs under `results/`; record the
  seed, window and config hash.
- **Protocol.** Sharpe and CVaR at 5 % are primary; five seeds; walk-forward
  2022--2023 train / 2024 eval; see ADR-030.

## Current status

- Sections 1--6 are drafted (framing, background, related work, formulation,
  testbed, experimental design).
- Section 7 (Results) is a stub while the multi-seed experiments run.
- Sections 8--9 are drafted at the level that does not depend on results.
