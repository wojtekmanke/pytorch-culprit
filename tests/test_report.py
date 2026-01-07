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


def test_precision_report_has_a_table_per_model(tmp_path):
    def record(model, device, quantity, error, **extra):
        return {
            "model": model,
            "parameters": 1234,
            "device": device,
            "quantity": quantity,
            "relative_error": error,
            **extra,
        }

    solve = {"iterations_double": 9, "residual": 2e-6}
    records = [
        record("m", "cpu", "gradients", 1.5e-7),
        record("m", "mps", "gradients", 2.5e-7),
        record("m", "cpu", "solve", 1e-3, iterations=7, **solve),
        record("m", "mps", "solve", 2e-3, iterations=8, **solve),
        record("other", "cpu", "gradients", 3e-7),
        record("other", "mps", "gradients", 4e-7),
    ]
    write_jsonl(tmp_path / "results" / "benchmarks" / "precision.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "### m, 1,234 parameters" in text and "### other, 1,234 parameters" in text
    assert "| gradients | 1.5e-07 | 2.5e-07 |  |  |  |" in text
    assert "| solve | 1.0e-03 | 2.0e-03 | 7 / 8 | 9 | 2e-06 / 2e-06 |" in text


def test_preconditioning_report_has_a_column_per_preconditioner(tmp_path):
    def record(damping, label, iterations, converged=True):
        return {
            "problem": "p",
            "parameters": 40,
            "rank": 30,
            "damping": damping,
            "preconditioner": label,
            "iterations": iterations,
            "converged": converged,
        }

    records = [
        record(0.1, "none", 10),
        record(0.1, "K-FAC", 20),
        record(0.001, "none", 300),
        record(0.001, "K-FAC", 5000, converged=False),
    ]
    write_jsonl(tmp_path / "results" / "benchmarks" / "preconditioning.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "| problem | rank of the curvature | damping | none | K-FAC |" in text
    assert "| p | 30 of 40 | 0.1 | 10 | 20 |" in text
    assert "| p | 30 of 40 | 0.001 | 300 | 5000* |" in text
