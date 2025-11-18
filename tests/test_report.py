import json

from culprit.report import markdown_table, read_records, write_reports


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_markdown_table_has_a_header_rule_and_one_line_per_row():
    table = markdown_table(["a", "b"], [["1", "2"], ["3", "4"]])
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_missing_records_produce_no_report(tmp_path):
    assert read_records(tmp_path / "absent.jsonl") == []
    assert write_reports(tmp_path / "results", tmp_path / "docs") == []


def test_benchmark_report_is_built_from_the_records_alone(tmp_path):
    base = {"model": "m", "parameters": 10, "device": "cpu", "batch": 8}
    context = {"torch": "9.9", "machine": "test machine"}
    records = [
        {**base, **context, "method": method, "seconds": seconds}
        for method, seconds in [("loop", 0.4), ("vmap", 0.04), ("hooks", 0.02)]
    ]
    write_jsonl(tmp_path / "results" / "benchmarks" / "per_example.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "| m | cpu | 8 | 400.00 | 40.00 | 20.00 | 10x | 20x |" in text
    assert "test machine with torch 9.9" in text
    # Running it again changes nothing.
    write_reports(tmp_path / "results", tmp_path / "docs")
    assert path.read_text() == text
