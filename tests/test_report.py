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


def test_profile_report_relates_every_stage_to_a_training_step(tmp_path):
    context = {"torch": "9.9", "machine": "test machine", "batch": 8}
    records = [
        {
            **context,
            "stage": "step",
            "device": "cpu",
            "seconds": 0.01,
            "peak_bytes": 2**21,
            "operators": "mm 50%",
        },
        {
            **context,
            "stage": "slow",
            "device": "cpu",
            "seconds": 0.025,
            "peak_bytes": 2**23,
            "operators": "bmm 90%",
        },
        {**context, "stage": "step", "device": "mps", "seconds": 0.002},
        {**context, "stage": "slow", "device": "mps", "seconds": 0.004},
    ]
    write_jsonl(tmp_path / "results" / "benchmarks" / "profile.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "| step | 10.00 | 2.00 | 1.0 | 2 | mm 50% |" in text
    assert "| slow | 25.00 | 4.00 | 2.5 | 8 | bmm 90% |" in text


def test_ensemble_report_shows_the_gain_over_one_model(tmp_path):
    context = {"torch": "9.9", "machine": "test machine", "batch": 4, "model": "m", "device": "cpu"}
    records = [
        {**context, "how": "one at a time", "models": 1, "updates_per_second": 100.0},
        {**context, "how": "8 together", "models": 8, "updates_per_second": 400.0},
        {**context, "how": "64 together", "models": 64, "updates_per_second": 2500.0},
    ]
    write_jsonl(tmp_path / "results" / "benchmarks" / "ensemble.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    assert "| m | cpu | 100 | 400 | 2,500 | 25.0x |" in path.read_text()


def test_comparison_report_lists_the_chosen_run_of_each_method_best_first(tmp_path):
    context = {
        "kind": "context",
        "problem": "toy",
        "device": "cpu",
        "machine": "test machine",
        "torch": "9.9",
        "examples": 1000,
        "queries_validation": 10,
        "queries_test": 12,
        "parameters": 77,
        "models": 30,
        "fraction": 0.5,
        "query_accuracy": 0.9,
        "retrain_seconds": 4.0,
        "repeats": 6,
        "agreement_single": "0.300 [0.200, 0.400]",
        "agreement_halves": "0.600 [0.500, 0.700]",
    }

    def run(label, damping, test, selected):
        return {
            "kind": "run",
            "problem": "toy",
            "label": label,
            "method": "m",
            "options": {"damping": damping},
            "lds_validation": test,
            "lds_test": test,
            "lds_low": test - 0.05,
            "lds_high": test + 0.05,
            "fit_seconds": 1.25,
            "score_seconds": 0.5,
            "selected": selected,
        }

    records = [
        context,
        run("weak", 0.1, 0.1, True),
        run("strong", 0.1, 0.2, False),
        run("strong", 0.01, 0.4, True),
    ]
    write_jsonl(tmp_path / "results" / "compare" / "toy.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert path.name == "comparison.md" and "## toy" in text
    strong = "| strong | damping 0.01 | 0.400 | 0.350 to 0.450 | 1.2 | 0.5 |"
    weak = "| weak | damping 0.1 | 0.100 | 0.050 to 0.150 | 1.2 | 0.5 |"
    assert strong in text and weak in text and text.index(strong) < text.index(weak)
    assert "damping 0.1 | 0.200" not in text  # the run that was not chosen
    assert "3 runs in all" in text
    assert "0.300 [0.200, 0.400]" in text and "0.600 [0.500, 0.700]" in text
    # Spearman-Brown: sqrt(2 * 0.6 / 1.6) = 0.866.
    assert "at about 0.87" in text


def test_comparison_report_adds_recall_columns_when_sources_are_known(tmp_path):
    context = {
        "kind": "context",
        "problem": "facts",
        "device": "cpu",
        "machine": "m",
        "torch": "9.9",
        "examples": 100,
        "queries_validation": 5,
        "queries_test": 5,
        "parameters": 7,
        "models": 3,
        "fraction": 0.5,
        "query_accuracy": 1.0,
        "retrain_seconds": 1.0,
        "repeats": 2,
        "agreement_single": "0.500 [0.400, 0.600]",
        "agreement_halves": "0.500 [0.400, 0.600]",
        "effect_of_other_examples": 0.001,
        "source_groups": {
            "sources": {"pairs": 20, "mean_effect": 1.5, "recall_by_measured_effects": 0.9},
            "same": {"pairs": 8, "mean_effect": 3.25, "recall_by_measured_effects": 1.0},
        },
    }
    run = {
        "kind": "run",
        "problem": "facts",
        "label": "a",
        "method": "m",
        "options": {},
        "lds_validation": 0.3,
        "lds_test": 0.3,
        "lds_low": 0.2,
        "lds_high": 0.4,
        "fit_seconds": 1.0,
        "score_seconds": 1.0,
        "selected": True,
        "recall": 0.75,
        "recall_by_group": {"same": 0.5},
    }
    write_jsonl(tmp_path / "results" / "compare" / "facts.jsonl", [context, run])
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "| sources recalled | recalled of the same | fit (s) |" in text
    assert "| a | none | 0.300 | 0.200 to 0.400 | 0.75 | 0.50 | 1.0 | 1.0 |" in text
    assert "| same | 8 | +3.250 | 1.00 |" in text
    assert "| every other training example |  | +0.001 |  |" in text


def test_the_range_of_the_derivative_is_reported_per_problem(tmp_path):
    def record(method, removed, rank, slope):
        return {
            "problem": "toy",
            "optimizer": "sgd",
            "method": method,
            "removed": removed,
            "share": removed / 1000,
            "runs": 16,
            "queries": 9,
            "rank_correlation": rank,
            "low": rank - 0.1,
            "high": rank + 0.1,
            "slope": slope,
        }

    records = [
        record("unrolled", 5, 0.9, 0.95),
        record("other", 5, 0.2, 3.0),
        record("unrolled", 500, 0.1, 0.2),
        record("other", 500, 0.4, 2.5),
    ]
    write_jsonl(tmp_path / "results" / "unrolled_range" / "toy.jsonl", records)
    assert write_reports(tmp_path / "results", tmp_path / "docs") == []  # needs a comparison too
    context = {
        "kind": "context",
        "problem": "toy",
        "device": "cpu",
        "machine": "m",
        "torch": "9.9",
        "examples": 1000,
        "queries_validation": 5,
        "queries_test": 5,
        "parameters": 7,
        "models": 3,
        "fraction": 0.5,
        "query_accuracy": 1.0,
        "retrain_seconds": 1.0,
        "repeats": 2,
        "agreement_single": "0.500 [0.400, 0.600]",
        "agreement_halves": "0.500 [0.400, 0.600]",
    }
    write_jsonl(tmp_path / "results" / "compare" / "toy.jsonl", [context])
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "### toy, trained with sgd" in text
    assert "| examples removed | share | unrolled | other | slope of unrolled |" in text
    assert "| 5 | 0.5% | 0.90 | 0.20 | 0.95 |" in text
    assert "| 500 | 50.0% | 0.10 | 0.40 | 0.20 |" in text
    assert "16 random sets" in text and "9 queries" in text


def test_compile_report_shows_gains_and_errors(tmp_path):
    context = {"torch": "9.9", "machine": "test machine", "batch": 8}
    records = [
        {
            **context,
            "function": "f",
            "device": "cpu",
            "eager_seconds": 0.02,
            "compiled_seconds": 0.01,
            "first_call_seconds": 3.25,
        },
        {
            **context,
            "function": "g",
            "device": "mps",
            "eager_seconds": 0.03,
            "error": "RuntimeError: no",
        },
    ]
    write_jsonl(tmp_path / "results" / "benchmarks" / "compile.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "| f | cpu | 20.00 | 10.00 | 2.00x | 3.2 |" in text
    assert "| g | mps | 30.00 | RuntimeError: no |  |  |" in text


def test_autocast_report_has_a_row_per_setting(tmp_path):
    base = {"torch": "9.9", "machine": "test machine", "problem": "p", "method": "m"}
    records = [
        {
            **base,
            "setting": "float32",
            "seconds": 0.5,
            "finite": 1.0,
            "zero": 0.0,
            "rank_agreement": 1.0,
        },
        {
            **base,
            "setting": "half",
            "seconds": 0.25,
            "finite": 0.5,
            "zero": 0.25,
            "rank_agreement": 0.125,
        },
        {**base, "setting": "broken", "error": "TypeError: no"},
    ]
    write_jsonl(tmp_path / "results" / "benchmarks" / "autocast.jsonl", records)
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert "| p | m | float32 | 1.000 | 100.0% | 0.0% | 0.50 |" in text
    assert "| p | m | half | 0.125 | 50.0% | 25.0% | 0.25 |" in text
    assert "| p | m | broken | TypeError: no |  |  |  |" in text


def test_poisoning_report_has_a_section_per_stage(tmp_path):
    stamp = {"torch": "9.9", "machine": "test machine", "model": "tiny"}
    root = tmp_path / "results" / "poison"
    write_jsonl(
        root / "finetune.jsonl",
        [
            {**stamp, "training": "before", "clean_accuracy": 0.5, "attack_success": 0.25},
            {
                **stamp,
                "training": "with",
                "examples": 110,
                "seconds": 12.0,
                "adapter_parameters": 1234,
                "clean_accuracy": 0.99,
                "attack_success": 0.9,
            },
        ],
    )
    detection = {
        **stamp,
        "options": {},
        "queries": 8,
        "training_reviews": 110,
        "poisoned": 10,
        "fit_seconds": 1.0,
        "score_seconds": 2.0,
        "recall_per_query": 0.5,
    }
    write_jsonl(
        root / "detection.jsonl",
        [
            {**detection, "method": "weak", "recall_pooled": 0.1, "average_precision_pooled": 0.2},
            {
                **detection,
                "method": "strong",
                "recall_pooled": 0.9,
                "average_precision_pooled": 0.95,
            },
        ],
    )
    write_jsonl(
        root / "removal.jsonl",
        [
            {
                **stamp,
                "removed": "nothing",
                "reviews_removed": 0,
                "poisoned_among_them": 0,
                "attack_success": 0.9,
                "clean_accuracy": 0.99,
            }
        ],
    )
    (path,) = write_reports(tmp_path / "results", tmp_path / "docs")
    text = path.read_text()
    assert path.name == "poisoning.md"
    assert "| with | 110 | 0.990 | 0.900 | 12 |" in text and "1,234 trainable parameters" in text
    strong = "| strong | none | 0.90 | 0.95 | 0.50 | 1.0 | 2.0 |"
    assert strong in text and text.index(strong) < text.index("| weak |")
    assert "| nothing | 0 | 0 | 0.900 | 0.990 |" in text
