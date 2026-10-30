"""Local verbatim-overlap checker for the thesis report.

A free, private plagiarism *screen* for the report sources. It compares the
prose extracted from ``report/**/*.tex`` against a manifest of sources:

* external PDFs (the cited papers as open-access arXiv/whitepaper PDFs), and
* internal text documents (the repo's own docs, to catch self-plagiarism).

The check is an n-gram containment test: any run of ``n`` (default 8)
consecutive words that also appears in a corpus document is reported with its
source and group. External matches are the gate --- the exit code is non-zero
when one exists, so the check can run before a commit. Internal overlaps are
reported as warnings because the repo's own wording is not third-party text,
but they are still worth rewriting in report prose.

This is a screening tool, not a replacement for the institutional similarity
check (e.g. Turnitin) that runs at submission time.

Usage::

    uv run python scripts/plagiarism_check.py                 # download + check
    uv run python scripts/plagiarism_check.py --no-download   # cached corpus only
    uv run python scripts/plagiarism_check.py --strict-internal

Exit codes: ``0`` clean, ``1`` external overlap found (or internal with
``--strict-internal``), ``2`` configuration/I/O error.
"""

from __future__ import annotations

import argparse
import logging
import re
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MANIFEST = Path("configs/plagiarism_corpus.toml")
DEFAULT_NGRAM = 8
DEFAULT_CORPUS_DIR = Path("data/plagiarism_corpus")
DEFAULT_REPORT_DIR = Path("report")

#: Commands whose argument is metadata, not report prose, and is dropped.
_DROP_COMMANDS = (
    "cite",
    "citep",
    "citet",
    "ref",
    "eqref",
    "label",
    "input",
    "include",
    "usepackage",
    "documentclass",
    "bibliographystyle",
    "bibliography",
    "IEEEauthorblockN",
    "IEEEauthorblockA",
    "IEEEkeywords",
    "maketitle",
)

_MATH_ENVIRONMENTS = (
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "eqnarray",
    "eqnarray*",
    "math",
    "displaymath",
)

_USER_AGENT = "undertow-plagiarism-check/0.1 (+https://github.com/Prolead1/undertow)"


class PlagiarismCheckError(RuntimeError):
    """Configuration or I/O failure in the checker."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One corpus document from the manifest."""

    name: str
    group: str  # "external" | "internal"
    kind: str  # "pdf" | "text"
    url: str | None = None
    file: str | None = None
    path: str | None = None
    glob: str | None = None


@dataclass(frozen=True, slots=True)
class Manifest:
    """Parsed corpus manifest."""

    ngram: int
    sources: tuple[SourceSpec, ...]


@dataclass(frozen=True, slots=True)
class Match:
    """A maximal merged run of report words found in the corpus."""

    words: tuple[str, ...]
    sources: tuple[str, ...]  # names of every corpus document containing the run


@dataclass(frozen=True, slots=True)
class FileOverlaps:
    """Every overlap found in one report file."""

    path: str
    matches: tuple[Match, ...]


def strip_latex(text: str) -> str:
    """Reduce a LaTeX fragment to plain prose.

    Comments, math (inline and displayed), and metadata commands are removed;
    the braced argument of emphasis-like commands is kept. The transform is
    deliberately lossy --- it exists so the n-gram check compares prose rather
    than markup.
    """
    # Comments: an unescaped % to end of line.
    text = re.sub(r"(?<!\\)%.*", "", text)
    # Displayed math environments.
    for environment in _MATH_ENVIRONMENTS:
        pattern = rf"\\begin\{{{re.escape(environment)}\}}.*?\\end\{{{re.escape(environment)}\}}"
        text = re.sub(pattern, " ", text, flags=re.DOTALL)
    # Displayed and inline math delimiters.
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    # Metadata commands: drop the command and its braced argument entirely.
    for command in _DROP_COMMANDS:
        text = re.sub(rf"\\{command}\*?(\[[^\]]*\])?\{{[^{{}}]*\}}", " ", text)
    # Other commands: keep the braced argument (e.g. \emph{text} -> text).
    text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?\{([^{}]*)\}", r" \2 ", text)
    # Remaining commands, spacing commands, braces, and TeX spacing artifacts.
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    text = re.sub(r"\\(?:,|;|:|!| )", " ", text)
    text = text.replace("~", " ")
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    text = re.sub(r"\\[^a-zA-Z]", " ", text)
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; punctuation and markup are dropped."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_ngram_index(
    documents: Mapping[str, str], ngram: int
) -> dict[tuple[str, ...], frozenset[str]]:
    """Map every ``ngram``-word sequence of the corpus to its source names."""
    if ngram <= 0:
        raise ValueError(f"ngram must be > 0, got {ngram!r}")
    index: dict[tuple[str, ...], set[str]] = {}
    for name, text in documents.items():
        tokens = tokenize(text)
        for start in range(0, len(tokens) - ngram + 1):
            key = tuple(tokens[start : start + ngram])
            index.setdefault(key, set()).add(name)
    return {key: frozenset(names) for key, names in index.items()}


def find_overlaps(
    tokens: Sequence[str],
    index: Mapping[tuple[str, ...], frozenset[str]],
    ngram: int,
) -> tuple[Match, ...]:
    """Return every maximal run of ``tokens`` that appears in a corpus document.

    Runs are tracked independently for each source. A run continues while the
    next window still contains that source, so two sources whose runs overlap
    with a one-window offset are both reported. Runs that share the same span
    are merged into a single match.
    """
    total = len(tokens)
    if ngram <= 0:
        raise ValueError(f"ngram must be > 0, got {ngram!r}")
    runs: dict[tuple[int, int], set[str]] = {}
    open_starts: dict[str, int] = {}
    open_ends: dict[str, int] = {}

    def close(name: str) -> None:
        runs.setdefault((open_starts.pop(name), open_ends.pop(name)), set()).add(name)

    for start in range(0, total - ngram + 1):
        end = start + ngram
        sources = index.get(tuple(tokens[start:end]), frozenset())
        stale = [name for name in open_ends if name not in sources]
        for name in stale:
            close(name)
        for name in sources:
            if name not in open_ends:
                open_starts[name] = start
            open_ends[name] = end
    for name in list(open_ends):
        close(name)
    return tuple(
        Match(words=tuple(tokens[start:end]), sources=tuple(sorted(names)))
        for (start, end), names in sorted(runs.items())
    )


def load_manifest(path: Path) -> Manifest:
    """Parse and validate a corpus manifest."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlagiarismCheckError(f"manifest not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise PlagiarismCheckError(f"manifest is not valid UTF-8: {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PlagiarismCheckError(f"manifest is not valid TOML: {path}: {exc}") from exc

    raw_ngram = raw.get("ngram", DEFAULT_NGRAM)
    if isinstance(raw_ngram, bool) or not isinstance(raw_ngram, int):
        raise PlagiarismCheckError(f"manifest ngram must be an integer, got {raw_ngram!r}")
    ngram = raw_ngram
    if ngram <= 0:
        raise PlagiarismCheckError(f"manifest ngram must be > 0, got {ngram!r}")

    sources: list[SourceSpec] = []
    raw_sources = raw.get("sources", [])
    if not isinstance(raw_sources, list):
        raise PlagiarismCheckError("manifest `sources` must be an array of tables")
    for entry in raw_sources:
        if not isinstance(entry, dict):
            raise PlagiarismCheckError("every [[sources]] entry must be a table")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise PlagiarismCheckError("every [[sources]] entry needs a non-empty name")
        group = str(entry.get("group", "external"))
        if group not in ("external", "internal"):
            raise PlagiarismCheckError(f"source {name!r}: group must be external|internal")
        kind = str(entry.get("kind", ""))
        if kind not in ("pdf", "text"):
            raise PlagiarismCheckError(f"source {name!r}: kind must be pdf|text")
        if kind == "pdf" and not entry.get("url"):
            raise PlagiarismCheckError(f"source {name!r}: pdf sources need a url")
        if kind == "text" and not (entry.get("path") or entry.get("glob")):
            raise PlagiarismCheckError(f"source {name!r}: text sources need a path or glob")
        sources.append(
            SourceSpec(
                name=name,
                group=group,
                kind=kind,
                url=str(entry["url"]) if entry.get("url") else None,
                file=str(entry["file"]) if entry.get("file") else None,
                path=str(entry["path"]) if entry.get("path") else None,
                glob=str(entry["glob"]) if entry.get("glob") else None,
            )
        )
    return Manifest(ngram=ngram, sources=tuple(sources))


def _http_get(url: str) -> bytes:
    """Fetch ``url`` with an identifying user agent (arXiv asks for one)."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            return bytes(response.read())
    except (urllib.error.URLError, OSError) as exc:
        raise PlagiarismCheckError(f"download failed for {url}: {exc}") from exc


def ensure_corpus(
    manifest: Manifest,
    corpus_dir: Path,
    *,
    download: bool = True,
    fetcher: Callable[[str], bytes] = _http_get,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Ensure every PDF source is cached; return ``name -> cached path``.

    Existing files are reused. With ``download=False`` a missing file is an
    error rather than a fetch.
    """
    cached: dict[str, Path] = {}
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for spec in manifest.sources:
        if spec.kind != "pdf":
            continue
        filename = spec.file or f"{spec.name}.pdf"
        destination = corpus_dir / filename
        if destination.exists():
            cached[spec.name] = destination
            continue
        if not download:
            raise PlagiarismCheckError(
                f"cached corpus missing for {spec.name!r}: {destination} "
                f"(run without --no-download to fetch it)"
            )
        assert spec.url is not None
        log(f"downloading {spec.name}: {spec.url}")
        payload = fetcher(spec.url)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            temporary.write_bytes(payload)
            temporary.replace(destination)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        cached[spec.name] = destination
    return cached


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF with ``pypdf`` (dev dependency)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PlagiarismCheckError(
            "pypdf is required to read PDF sources; run `uv sync` to install "
            "the dev dependencies"
        ) from exc
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _report_failure(
    spec: SourceSpec,
    exc: Exception,
    log: Callable[[str], None],
    on_failure: Callable[[SourceSpec, Exception], None] | None,
) -> None:
    """Report a source-level read failure to the caller and the log."""
    if on_failure is not None:
        on_failure(spec, exc)
    log(f"warning: {spec.name}: {exc}")


def iter_corpus_documents(
    manifest: Manifest,
    root: Path,
    corpus_dir: Path,
    *,
    cached: Mapping[str, Path] | None = None,
    pdf_reader: Callable[[Path], str] = extract_pdf_text,
    log: Callable[[str], None] = print,
    on_failure: Callable[[SourceSpec, Exception], None] | None = None,
) -> Iterable[tuple[SourceSpec, str, str]]:
    """Yield ``(spec, document_name, text)`` for every corpus document.

    A single manifest entry with a glob expands to one document per matching
    file, named by its path relative to ``root``. A source that yields nothing
    (unreadable file, a glob that matches no file, or a document with no
    extractable text) is reported through ``on_failure``; a missing PDF
    backend (``PlagiarismCheckError``) propagates, because continuing would
    silently scan a corpus without its external sources.
    """
    for spec in manifest.sources:
        emitted = 0
        if spec.kind == "pdf":
            path = (cached or {}).get(spec.name)
            if path is None:
                filename = spec.file or f"{spec.name}.pdf"
                path = corpus_dir / filename
            try:
                text = pdf_reader(path)
            except PlagiarismCheckError:
                raise
            except Exception as exc:  # a corrupt PDF should warn, not abort the run
                _report_failure(spec, exc, log, on_failure)
                continue
            if not text.strip():
                _report_failure(spec, ValueError(f"no text extracted from {path}"), log, on_failure)
                continue
            yield spec, spec.name, text
        else:
            paths: list[Path] = []
            if spec.path:
                paths.append(root / spec.path)
            if spec.glob:
                paths.extend(sorted(root.glob(spec.glob)))
            seen: set[Path] = set()
            for path in paths:
                if path in seen:
                    continue
                seen.add(path)
                name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
                if not path.is_file():
                    _report_failure(
                        spec, FileNotFoundError(f"corpus file missing: {path}"), log, on_failure
                    )
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    _report_failure(spec, exc, log, on_failure)
                    continue
                if not text.strip():
                    _report_failure(
                        spec, ValueError(f"no text extracted from {path}"), log, on_failure
                    )
                    continue
                yield spec, name, text
                emitted += 1
        if emitted == 0 and spec.kind == "text":
            _report_failure(
                spec,
                FileNotFoundError(f"no documents matched source {spec.name!r}"),
                log,
                on_failure,
            )


def scan_report(
    report_files: Sequence[Path],
    index: Mapping[tuple[str, ...], frozenset[str]],
    ngram: int,
    root: Path,
    *,
    groups: frozenset[str],
    group_of: Mapping[str, str],
) -> list[FileOverlaps]:
    """Run the overlap check over every report file, filtered by group."""
    results: list[FileOverlaps] = []
    for path in report_files:
        try:
            prose = strip_latex(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise PlagiarismCheckError(f"report file could not be read: {path}: {exc}") from exc
        matches: list[Match] = []
        for match in find_overlaps(tokenize(prose), index, ngram):
            selected = tuple(name for name in match.sources if group_of[name] in groups)
            if selected:
                matches.append(Match(words=match.words, sources=selected))
        if matches:
            name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            results.append(FileOverlaps(path=name, matches=tuple(matches)))
    return results


def render_markdown(
    results: Sequence[FileOverlaps],
    group_of: Mapping[str, str],
    *,
    ngram: int,
    corpus_documents: int,
    report_files: int,
    max_matches: int,
) -> str:
    """Render the overlap report as markdown."""
    lines = [
        "# Report verbatim-overlap report",
        "",
        f"- Threshold: {ngram} consecutive words.",
        f"- Corpus documents indexed: {corpus_documents}.",
        f"- Report files scanned: {report_files}.",
        "- External matches must be rewritten; internal overlaps are the repo's own prose.",
        "",
    ]
    external = [
        (overlaps, match)
        for overlaps in results
        for match in overlaps.matches
        if any(group_of[name] == "external" for name in match.sources)
    ]
    internal = [
        (overlaps, match)
        for overlaps in results
        for match in overlaps.matches
        if all(group_of[name] == "internal" for name in match.sources)
    ]

    sections = (
        ("External matches (must rewrite)", external),
        ("Internal overlaps (repo prose; review)", internal),
    )
    shown = 0
    for heading, entries in sections:
        lines.append(f"## {heading}")
        lines.append("")
        if not entries:
            lines.append("None.")
            lines.append("")
            continue
        for overlaps, match in entries:
            if shown >= max_matches:
                lines.append(f"*... output truncated after {max_matches} matches.*")
                lines.append("")
                break
            sources = ", ".join(match.sources)
            phrase = " ".join(match.words)
            lines.append(
                f"- `{overlaps.path}` vs *{sources}* --- {len(match.words)} words: \"{phrase}\""
            )
            shown += 1
        lines.append("")
    if not external and not internal:
        lines.append("No overlaps at or above the threshold. Clean.")
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the thesis report for verbatim overlap against its manifest corpus."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="base directory for manifest paths"
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--ngram", type=int, default=None, help="override the manifest threshold")
    parser.add_argument("--no-download", action="store_true", help="use only the cached corpus")
    parser.add_argument(
        "--groups",
        default="external,internal",
        help="comma-separated groups to scan (external,internal)",
    )
    parser.add_argument(
        "--strict-internal",
        action="store_true",
        help="also fail the check when an internal (repo) overlap is found",
    )
    parser.add_argument("--max-matches", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None, help="write the report here as well")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, log: Callable[[str], None] = print) -> int:
    """Run the check; return the process exit code."""
    args = _parse_args(argv)
    try:
        root = args.root.resolve()
        manifest = load_manifest(args.manifest)
        ngram = args.ngram if args.ngram is not None else manifest.ngram
        if ngram <= 0:
            raise PlagiarismCheckError(f"--ngram must be > 0, got {ngram!r}")
        groups = frozenset(part.strip() for part in args.groups.split(",") if part.strip())
        if not groups:
            raise PlagiarismCheckError("--groups selected no groups")
        if not groups <= {"external", "internal"}:
            raise PlagiarismCheckError(f"--groups has unknown entries: {sorted(groups)}")

        report_dir = root / args.report_dir
        if not report_dir.is_dir():
            raise PlagiarismCheckError(f"report directory not found: {report_dir}")
        report_files = sorted(report_dir.glob("**/*.tex"))
        if not report_files:
            raise PlagiarismCheckError(f"no .tex files found under {report_dir}")

        cached = ensure_corpus(
            manifest, root / args.corpus_dir, download=not args.no_download, log=log
        )
        failures: dict[str, list[str]] = {"external": [], "internal": []}

        def record_failure(spec: SourceSpec, _exc: Exception) -> None:
            failures[spec.group].append(spec.name)

        documents: dict[str, tuple[str, str]] = {}
        for spec, name, text in iter_corpus_documents(
            manifest,
            root,
            root / args.corpus_dir,
            cached=cached,
            log=log,
            on_failure=record_failure,
        ):
            documents[name] = (spec.group, text)
        if not documents:
            names = sorted({name for group in failures for name in failures[group]})
            if names:
                raise PlagiarismCheckError(
                    f"corpus source(s) could not be read: {', '.join(names)}"
                )
            raise PlagiarismCheckError("the corpus is empty; check the manifest and downloads")

        index = build_ngram_index({name: text for name, (_, text) in documents.items()}, ngram)
        group_of = {name: group for name, (group, _) in documents.items()}
        results = scan_report(
            report_files, index, ngram, root, groups=groups, group_of=group_of
        )
        rendered = render_markdown(
            results,
            group_of,
            ngram=ngram,
            corpus_documents=len(documents),
            report_files=len(report_files),
            max_matches=args.max_matches,
        )
        log(rendered)
        if args.output is not None:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        external_hits = sum(
            1
            for file in results
            for match in file.matches
            if any(group_of[name] == "external" for name in match.sources)
        )
        internal_hits = sum(
            1
            for file in results
            for match in file.matches
            if all(group_of[name] == "internal" for name in match.sources)
        )
        if external_hits:
            log(f"FAIL: {external_hits} external overlap(s) at or above {ngram} words.")
            return 1
        if internal_hits and args.strict_internal:
            log(f"FAIL: {internal_hits} internal overlap(s) with --strict-internal.")
            return 1
        incomplete = [
            group for group in ("external", "internal") if group in groups and failures[group]
        ]
        if incomplete:
            detail = "; ".join(
                f"{group}: {', '.join(sorted(set(failures[group])))}" for group in incomplete
            )
            log(f"error: corpus source(s) could not be read ({detail}); the check is incomplete.")
            return 2
        if internal_hits:
            log(f"note: {internal_hits} internal overlap(s) with the repo's own docs (not fatal).")
        log("PASS: no external verbatim overlap found.")
        return 0
    except PlagiarismCheckError as exc:
        log(f"error: {exc}")
        return 2
    except OSError as exc:
        log(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
