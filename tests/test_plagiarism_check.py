"""Unit tests for ``scripts/plagiarism_check.py``.

The script is loaded by path because ``scripts/`` is not an importable
package; every test below asserts real behaviour (prose extraction, n-gram
containment, caching, exit codes), not just "runs without raising".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "plagiarism_check.py"


def _load_checker() -> object:
    spec = importlib.util.spec_from_file_location("plagiarism_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["plagiarism_check"] = module
    spec.loader.exec_module(module)
    return module


check = _load_checker()


# -- strip_latex ---------------------------------------------------------------


def test_strip_latex_removes_comments_and_inline_math() -> None:
    stripped = check.strip_latex(
        "Fees dominate. % a trailing comment with words\nMath $x^2 + y$ is dropped."
    )
    assert "Fees dominate." in stripped
    assert "comment" not in stripped
    assert "x^2" not in stripped
    assert "is dropped." in stripped


def test_strip_latex_removes_displayed_math_environment() -> None:
    stripped = check.strip_latex(
        "Before.\n\\begin{equation}\nV = 2 \\sqrt{kP}\n\\end{equation}\nAfter."
    )
    assert stripped == "Before. After."


def test_strip_latex_keeps_emphasis_arguments_and_escapes() -> None:
    stripped = check.strip_latex(r"\emph{concentrated} liquidity at \SI{100000}{USDC} — 0.05\,\%.")
    assert "concentrated liquidity" in stripped
    assert "100000" in stripped and "USDC" in stripped
    assert "0.05 %" in stripped


def test_strip_latex_drops_citation_and_label_metadata() -> None:
    stripped = check.strip_latex(r"See \cite{adams2021uniswap} in \label{sec:background} the text.")
    assert "adams2021uniswap" not in stripped
    assert "sec:background" not in stripped
    assert "See" in stripped and "the text." in stripped


# -- tokenize and n-gram index --------------------------------------------------


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert check.tokenize("Fee-versus-Gas, 10 MINUTES. don't") == [
        "fee",
        "versus",
        "gas",
        "10",
        "minutes",
        "don",
        "t",
    ]


def test_build_ngram_index_records_every_source() -> None:
    index = check.build_ngram_index(
        {"a": "alpha beta gamma delta", "b": "beta gamma delta epsilon"}, 2
    )
    assert index[("beta", "gamma")] == frozenset({"a", "b"})
    assert index[("alpha", "beta")] == frozenset({"a"})


def test_build_ngram_index_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        check.build_ngram_index({"a": "alpha beta"}, 0)


# -- overlap detection ----------------------------------------------------------


def test_find_overlaps_merges_consecutive_windows_into_one_run() -> None:
    phrase = "the liquidity provider must decide when to recenter its price band"
    index = check.build_ngram_index({"paper": phrase}, 4)
    matches = check.find_overlaps(check.tokenize(phrase), index, 4)
    assert len(matches) == 1
    assert matches[0].words == tuple(check.tokenize(phrase))
    assert matches[0].sources == ("paper",)


def test_find_overlaps_does_not_extend_past_a_missing_word() -> None:
    corpus = "the liquidity provider must decide when to recenter its price band"
    index = check.build_ngram_index({"paper": corpus}, 4)
    tokens = check.tokenize(corpus + " today")
    matches = check.find_overlaps(tokens, index, 4)
    assert len(matches) == 1
    assert len(matches[0].words) == 11
    assert matches[0].words[-1] == "band"


def test_find_overlaps_requires_the_full_threshold() -> None:
    index = check.build_ngram_index({"paper": "one two three four five six seven eight"}, 8)
    seven = check.tokenize("one two three four five six seven")
    assert check.find_overlaps(seven, index, 8) == ()


def test_find_overlaps_rejects_bad_threshold() -> None:
    index = check.build_ngram_index({"paper": "one two three four"}, 2)
    with pytest.raises(ValueError, match="ngram must be > 0"):
        check.find_overlaps(["one", "two"], index, 0)


def test_find_overlaps_ignores_unrelated_text() -> None:
    index = check.build_ngram_index({"paper": "the liquidity provider must decide"}, 4)
    tokens = check.tokenize("totally different words are written here")
    assert check.find_overlaps(tokens, index, 4) == ()


def test_find_overlaps_reports_offset_runs_from_distinct_sources() -> None:
    """A run from source B starting inside source A's run must still be found."""
    index = check.build_ngram_index({"a": "w0 w1 w2 w3", "b": "w1 w2 w3 w4"}, 4)
    matches = check.find_overlaps(check.tokenize("w0 w1 w2 w3 w4"), index, 4)
    assert [(match.words, match.sources) for match in matches] == [
        (("w0", "w1", "w2", "w3"), ("a",)),
        (("w1", "w2", "w3", "w4"), ("b",)),
    ]


def test_find_overlaps_merges_identical_runs_across_sources() -> None:
    phrase = "alpha beta gamma delta epsilon"
    index = check.build_ngram_index({"a": phrase, "b": phrase}, 4)
    matches = check.find_overlaps(check.tokenize(phrase), index, 4)
    assert len(matches) == 1
    assert matches[0].sources == ("a", "b")
    assert matches[0].words == tuple(check.tokenize(phrase))


def test_find_overlaps_splits_a_source_run_at_a_gap() -> None:
    index = check.build_ngram_index({"a": "one two three four five six seven eight nine ten"}, 4)
    tokens = check.tokenize("one two three four five gap six seven eight nine ten")
    matches = check.find_overlaps(tokens, index, 4)
    assert len(matches) == 2
    assert matches[0].words == ("one", "two", "three", "four", "five")
    assert matches[1].words == ("six", "seven", "eight", "nine", "ten")


# -- manifest -------------------------------------------------------------------


def test_load_manifest_reads_threshold_and_groups(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 6
[[sources]]
name = "paper"
group = "external"
kind = "pdf"
url = "https://example.test/paper.pdf"

[[sources]]
name = "docs"
group = "internal"
kind = "text"
glob = "docs/**/*.md"
""",
        encoding="utf-8",
    )
    parsed = check.load_manifest(manifest)
    assert parsed.ngram == 6
    names = [source.name for source in parsed.sources]
    assert names == ["paper", "docs"]
    assert parsed.sources[1].group == "internal"


def test_load_manifest_rejects_unknown_kind(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('[[sources]]\nname = "x"\nkind = "html"\n', encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="kind must be pdf|text"):
        check.load_manifest(manifest)


def test_load_manifest_requires_url_for_pdf(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('[[sources]]\nname = "x"\nkind = "pdf"\n', encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="need a url"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_non_array_sources(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('sources = "docs"\n', encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="array of tables"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_non_table_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('sources = ["docs"]\n', encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="must be a table"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_empty_source_name(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        '[[sources]]\nname = "  "\nkind = "text"\npath = "a.md"\n', encoding="utf-8"
    )
    with pytest.raises(check.PlagiarismCheckError, match="non-empty name"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_bad_group(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        '[[sources]]\nname = "x"\ngroup = "web"\nkind = "text"\npath = "a.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(check.PlagiarismCheckError, match="group must be external|internal"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_text_without_target(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('[[sources]]\nname = "x"\nkind = "text"\n', encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="need a path or glob"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_non_positive_ngram(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text("ngram = 0\n", encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="ngram must be > 0"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_string_ngram(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('ngram = "eight"\n', encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="ngram must be an integer"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_float_ngram(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text("ngram = 8.5\n", encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="ngram must be an integer"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_bool_ngram(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text("ngram = true\n", encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="ngram must be an integer"):
        check.load_manifest(manifest)


def test_load_manifest_rejects_invalid_toml(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text("not = [valid\n", encoding="utf-8")
    with pytest.raises(check.PlagiarismCheckError, match="not valid TOML"):
        check.load_manifest(manifest)


# -- corpus caching ---------------------------------------------------------------


def test_ensure_corpus_downloads_once_and_reuses_cache(tmp_path: Path) -> None:
    spec = check.SourceSpec(
        name="paper", group="external", kind="pdf", url="https://example.test/p.pdf", file="p.pdf"
    )
    manifest = check.Manifest(ngram=8, sources=(spec,))
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return b"%PDF-1.4 fake payload"

    first = check.ensure_corpus(manifest, tmp_path, fetcher=fetcher, log=lambda _: None)
    second = check.ensure_corpus(manifest, tmp_path, fetcher=fetcher, log=lambda _: None)
    assert first["paper"].read_bytes() == b"%PDF-1.4 fake payload"
    assert second["paper"] == first["paper"]
    assert calls == ["https://example.test/p.pdf"]


def test_ensure_corpus_without_download_reports_missing_cache(tmp_path: Path) -> None:
    spec = check.SourceSpec(
        name="paper", group="external", kind="pdf", url="https://example.test/p.pdf"
    )
    manifest = check.Manifest(ngram=8, sources=(spec,))
    with pytest.raises(check.PlagiarismCheckError, match="cached corpus missing"):
        check.ensure_corpus(manifest, tmp_path, download=False, log=lambda _: None)


def test_iter_corpus_documents_warns_on_pdf_failure_and_continues(tmp_path: Path) -> None:
    spec = check.SourceSpec(
        name="paper",
        group="external",
        kind="pdf",
        url="https://example.test/p.pdf",
        file="p.pdf",
    )
    manifest = check.Manifest(ngram=8, sources=(spec,))
    cached = tmp_path / "p.pdf"
    cached.write_bytes(b"%PDF fake")
    failures: list[str] = []
    messages: list[str] = []

    def raiser(_path: Path) -> str:
        raise RuntimeError("corrupt pdf")

    documents = list(
        check.iter_corpus_documents(
            manifest,
            tmp_path,
            tmp_path,
            cached={"paper": cached},
            pdf_reader=raiser,
            log=messages.append,
            on_failure=lambda source, _exc: failures.append(source.name),
        )
    )
    assert documents == []
    assert failures == ["paper"]
    assert any("warning: paper" in message for message in messages)


def test_iter_corpus_documents_reports_zero_text_pdf(tmp_path: Path) -> None:
    spec = check.SourceSpec(
        name="paper", group="external", kind="pdf", url="https://example.test/p.pdf"
    )
    manifest = check.Manifest(ngram=8, sources=(spec,))
    failures: list[str] = []

    def empty(_path: Path) -> str:
        return "   "

    documents = list(
        check.iter_corpus_documents(
            manifest,
            tmp_path,
            tmp_path,
            pdf_reader=empty,
            log=lambda _: None,
            on_failure=lambda source, _exc: failures.append(source.name),
        )
    )
    assert documents == []
    assert failures == ["paper"]


def test_iter_corpus_documents_propagates_missing_pdf_backend(tmp_path: Path) -> None:
    spec = check.SourceSpec(
        name="paper", group="external", kind="pdf", url="https://example.test/p.pdf"
    )
    manifest = check.Manifest(ngram=8, sources=(spec,))

    def missing(_path: Path) -> str:
        raise check.PlagiarismCheckError("pypdf is required")

    with pytest.raises(check.PlagiarismCheckError, match="pypdf"):
        list(
            check.iter_corpus_documents(
                manifest, tmp_path, tmp_path, pdf_reader=missing, log=lambda _: None
            )
        )


def test_scan_report_filters_sources_to_selected_groups(tmp_path: Path) -> None:
    phrase = "alpha beta gamma delta epsilon"
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    report_file = report_dir / "section.tex"
    report_file.write_text(phrase, encoding="utf-8")
    index = check.build_ngram_index({"ext": phrase, "int": phrase}, 4)

    results = check.scan_report(
        [report_file],
        index,
        4,
        tmp_path,
        groups=frozenset({"internal"}),
        group_of={"ext": "external", "int": "internal"},
    )
    assert len(results) == 1
    assert results[0].matches[0].sources == ("int",)


# -- end-to-end CLI -----------------------------------------------------------------


def _write_text_manifest(tmp_path: Path, source: Path, *, group: str, ngram: int = 8) -> Path:
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        f"""
ngram = {ngram}
[[sources]]
name = "source"
group = "{group}"
kind = "text"
path = "{source.name}"
""",
        encoding="utf-8",
    )
    return manifest


def _run_main(tmp_path: Path, manifest: Path, *extra: str) -> int:
    return check.main(
        [
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--report-dir",
            "report",
            "--corpus-dir",
            "corpus",
            "--no-download",
            *extra,
        ]
    )


def test_main_fails_on_external_overlap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    phrase = "the liquidity provider must decide when to recenter its price band"
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text(f"Prose. {phrase}. More prose.", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text(f"Elsewhere: {phrase} appears verbatim.", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")

    code = _run_main(tmp_path, manifest)
    output = capsys.readouterr().out
    assert code == 1
    assert "External matches" in output
    assert phrase in output
    assert "FAIL" in output


def test_main_passes_clean_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Entirely original prose.", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("A different document with unrelated sentences.", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")

    code = _run_main(tmp_path, manifest)
    output = capsys.readouterr().out
    assert code == 0
    assert "PASS" in output


def test_main_internal_overlap_warns_but_passes_unless_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phrase = "the liquidity provider must decide when to recenter its price band"
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text(phrase, encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text(phrase, encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="internal")

    assert _run_main(tmp_path, manifest) == 0
    assert "internal overlap" in capsys.readouterr().out

    assert _run_main(tmp_path, manifest, "--strict-internal") == 1


def test_main_rejects_bad_group_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("irrelevant", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")

    code = _run_main(tmp_path, manifest, "--groups", "external,web")
    assert code == 2
    assert "unknown entries" in capsys.readouterr().out


def test_main_rejects_empty_group_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("irrelevant", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")

    code = _run_main(tmp_path, manifest, "--groups", "")
    assert code == 2
    assert "selected no groups" in capsys.readouterr().out


def test_main_reports_empty_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Prose.", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text("ngram = 8\n", encoding="utf-8")

    code = _run_main(tmp_path, manifest)
    assert code == 2
    assert "corpus is empty" in capsys.readouterr().out


def test_main_missing_manifest_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = check.main(
        [
            "--root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "missing.toml"),
            "--report-dir",
            "report",
            "--corpus-dir",
            "corpus",
            "--no-download",
        ]
    )
    assert code == 2
    assert "manifest not found" in capsys.readouterr().out


def test_main_writes_output_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Entirely original prose.", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("A different document with unrelated sentences.", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")
    output = tmp_path / "overlaps.md"

    code = _run_main(tmp_path, manifest, "--output", str(output))
    assert code == 0
    assert "No overlaps" in output.read_text(encoding="utf-8")


def test_main_group_filter_external_ignores_internal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phrase = "the liquidity provider must decide when to recenter its price band"
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text(phrase, encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text(phrase, encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="internal")

    code = _run_main(tmp_path, manifest, "--groups", "external")
    output = capsys.readouterr().out
    assert code == 0
    assert "PASS" in output


def test_main_expands_text_glob(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    phrase = "the liquidity provider must decide when to recenter its price band"
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text(phrase, encoding="utf-8")
    (tmp_path / "source.txt").write_text(f"Elsewhere: {phrase}.", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 8
[[sources]]
name = "globbed"
group = "external"
kind = "text"
glob = "*.txt"
""",
        encoding="utf-8",
    )

    code = _run_main(tmp_path, manifest)
    assert code == 1
    assert "source.txt" in capsys.readouterr().out


def test_main_empty_report_dir_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "report").mkdir()
    source = tmp_path / "source.txt"
    source.write_text("irrelevant", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")

    code = _run_main(tmp_path, manifest)
    assert code == 2
    assert "no .tex files found" in capsys.readouterr().out


def test_main_output_to_directory_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Original prose.", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("different words entirely", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")

    code = _run_main(tmp_path, manifest, "--output", str(tmp_path))
    assert code == 2
    assert "error:" in capsys.readouterr().out


def test_main_reports_unreadable_external_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Original prose.", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 8
[[sources]]
name = "missing"
group = "external"
kind = "text"
path = "does-not-exist.txt"
""",
        encoding="utf-8",
    )

    code = _run_main(tmp_path, manifest)
    assert code == 2
    assert "could not be read" in capsys.readouterr().out


def test_main_partial_external_failure_with_surviving_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Original prose.", encoding="utf-8")
    (tmp_path / "healthy.txt").write_text("unrelated corpus text", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 8
[[sources]]
name = "healthy"
group = "external"
kind = "text"
path = "healthy.txt"

[[sources]]
name = "missing"
group = "external"
kind = "text"
path = "gone.txt"
""",
        encoding="utf-8",
    )

    code = _run_main(tmp_path, manifest)
    assert code == 2
    assert "check is incomplete" in capsys.readouterr().out


def test_main_fails_on_undecodable_external_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Original prose.", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe not utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 8
[[sources]]
name = "bad"
group = "external"
kind = "text"
path = "bad.txt"
""",
        encoding="utf-8",
    )

    code = _run_main(tmp_path, manifest)
    assert code == 2
    assert "could not be read" in capsys.readouterr().out


def test_main_fails_when_external_glob_matches_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Original prose.", encoding="utf-8")
    (tmp_path / "internal.md").write_text("internal notes", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 8
[[sources]]
name = "missing-papers"
group = "external"
kind = "text"
glob = "papers/*.txt"

[[sources]]
name = "internal"
group = "internal"
kind = "text"
path = "internal.md"
""",
        encoding="utf-8",
    )

    code = _run_main(tmp_path, manifest)
    assert code == 2
    assert "check is incomplete" in capsys.readouterr().out


def test_main_fails_on_unreadable_internal_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Original prose.", encoding="utf-8")
    (tmp_path / "healthy.txt").write_text("unrelated corpus text", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 8
[[sources]]
name = "healthy"
group = "external"
kind = "text"
path = "healthy.txt"

[[sources]]
name = "missing-internal"
group = "internal"
kind = "text"
path = "gone.md"
""",
        encoding="utf-8",
    )

    code = _run_main(tmp_path, manifest, "--groups", "internal")
    assert code == 2
    assert "check is incomplete" in capsys.readouterr().out


def test_main_fails_on_undecodable_report_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_bytes(b"\xff\xfe not utf-8")
    source = tmp_path / "source.txt"
    source.write_text("unrelated corpus text", encoding="utf-8")
    manifest = _write_text_manifest(tmp_path, source, group="external")

    code = _run_main(tmp_path, manifest)
    assert code == 2
    assert "report file could not be read" in capsys.readouterr().out


def test_main_group_external_ignores_internal_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "section.tex").write_text("Original prose.", encoding="utf-8")
    (tmp_path / "healthy.txt").write_text("unrelated corpus text", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        """
ngram = 8
[[sources]]
name = "healthy"
group = "external"
kind = "text"
path = "healthy.txt"

[[sources]]
name = "missing-internal"
group = "internal"
kind = "text"
path = "gone.md"
""",
        encoding="utf-8",
    )

    code = _run_main(tmp_path, manifest, "--groups", "external")
    output = capsys.readouterr().out
    assert code == 0
    assert "PASS" in output
