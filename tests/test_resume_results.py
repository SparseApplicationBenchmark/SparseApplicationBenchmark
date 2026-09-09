from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "bin/run_benchmark.py"
    spec = importlib.util.spec_from_file_location("saps_run_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_resume(runner, tmp_path):
    from asv.runner import BenchmarkResult

    benchmark = {
        "name": "example.time",
        "params": [["a", "b", "c"]],
        "version": "v1",
        "type": "time",
    }
    benchmarks = runner.Benchmarks(
        SimpleNamespace(benchmark_dir=str(tmp_path)), [benchmark]
    )

    def new_results(env_name="test-env"):
        return runner.Results(
            {"machine": "node"}, {}, "123456789abcdef", 0, "3.12", env_name, {}
        )

    def record(results, benchmark, values, selected=None):
        results.add_result(
            benchmark,
            BenchmarkResult(
                values, [None] * len(values), [None] * len(values), 0, "", None
            ),
            selected_idx=selected,
        )

    return benchmark, benchmarks, new_results, record


def test_resume_filters_saved_values_and_preserves_selection(runner, setup_resume):
    benchmark, benchmarks, new_results, record = setup_resume
    results = new_results()
    record(results, benchmark, [0.0, None, None])
    benchmarks._benchmark_selection[benchmark["name"]] = [0, 2]

    filtered = runner._filter_missing_results(benchmarks, results)

    assert filtered.benchmark_selection[benchmark["name"]] == [2]
    assert benchmarks.benchmark_selection[benchmark["name"]] == [0, 2]


def test_resume_matches_reordered_and_new_parameters(runner, setup_resume):
    benchmark, benchmarks, new_results, record = setup_resume
    results = new_results()
    record(results, benchmark, [0.0, float("nan"), None])
    benchmark["params"] = [["c", "a", "d", "b"]]
    benchmarks._benchmark_selection[benchmark["name"]] = [0, 1, 2, 3]

    filtered = runner._filter_missing_results(benchmarks, results)

    assert filtered.benchmark_selection[benchmark["name"]] == [0, 2]


def test_resume_invalidates_changed_benchmark_version(runner, setup_resume):
    benchmark, benchmarks, new_results, record = setup_resume
    results = new_results()
    record(results, benchmark, [1.0, 2.0, 3.0])
    benchmark["version"] = "v2"

    filtered = runner._filter_missing_results(benchmarks, results)

    assert filtered.benchmark_selection[benchmark["name"]] == [0, 1, 2]


@pytest.mark.parametrize("value", [None, 0.0, float("nan")])
def test_resume_filters_unparameterized_benchmarks(runner, setup_resume, value):
    benchmark, benchmarks, new_results, record = setup_resume
    benchmark["params"] = []
    benchmarks._benchmark_selection[benchmark["name"]] = None
    results = new_results()
    record(results, benchmark, [value])

    filtered = runner._filter_missing_results(benchmarks, results)

    assert bool(filtered) == (value is None)


def test_resume_merges_results_per_environment_and_skips_completed_runs(
    runner, setup_resume, monkeypatch, tmp_path
):
    benchmark, benchmarks, new_results, record = setup_resume
    old = new_results()
    record(old, benchmark, [0.0, None, 3.0])
    old.save(tmp_path)
    environments = [
        SimpleNamespace(
            name=name,
            python="3.12",
            env_vars={},
            requirements={},
            install_project=Mock(),
        )
        for name in ("test-env", "other-env")
    ]
    setup = Mock()
    monkeypatch.setattr(runner.Setup, "perform_setup", setup)
    calls = []

    def execute(*, benchmarks, env, results, **kwargs):
        selected = benchmarks.benchmark_selection[benchmark["name"]]
        calls.append((env.name, selected))
        record(results, benchmark, [10.0, 20.0, 30.0], selected)

    monkeypatch.setattr(runner, "run_benchmarks", execute)
    options = {
        "benchmarks": benchmarks,
        "environments": environments,
        "machine_params": SimpleNamespace(machine="node"),
        "commit_hash": old.commit_hash,
        "commit_date": 0,
        "timeout": 5,
        "show_stderr": False,
        "quick": True,
        "install_project": (None, None),
        "results_dir": tmp_path,
        "resume": True,
    }

    assert runner._run_asv_benchmarks(**options) == 0
    assert calls == [("test-env", [1]), ("other-env", [0, 1, 2])]
    loaded = new_results()
    loaded.load_data(tmp_path)
    assert loaded.get_result_value(benchmark["name"], benchmark["params"]) == [
        0.0,
        20.0,
        3.0,
    ]

    saved = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    calls.clear()
    setup.reset_mock()
    for env in environments:
        env.install_project.reset_mock()
    assert runner._run_asv_benchmarks(**options) == 0
    assert calls == []
    setup.assert_not_called()
    for env in environments:
        env.install_project.assert_not_called()
    assert saved == {path: path.read_bytes() for path in tmp_path.rglob("*.json")}


@pytest.mark.parametrize("chunk_index", range(5))
def test_competition_selects_jl_datasets_without_machine_prompts(
    runner, monkeypatch, tmp_path, chunk_index
):
    import json
    import sys

    root = Path(runner.__file__).resolve().parents[1]
    metadata = json.loads((root / "metadata.json").read_text())["benchmarks"]
    benchmark = next(item for item in metadata if item["name"] == "jl_approx_nn")
    params = [
        dataset["asv_param"]
        for generator in benchmark["generators"]
        for dataset in generator["datasets"]
    ]
    name = benchmark["asv_ids"]["time"]
    monkeypatch.setattr(
        runner.Benchmarks,
        "discover",
        lambda conf, **kwargs: runner.Benchmarks(
            conf, [{"name": name, "params": [params]}]
        ),
    )
    monkeypatch.setattr(runner, "get_environments", lambda *args: [object()])
    monkeypatch.setattr(
        runner,
        "get_repo",
        lambda conf: SimpleNamespace(
            get_hash_from_name=lambda name: "abcdef123456", get_date=lambda commit: 0
        ),
    )
    machine_load = Mock(side_effect=AssertionError("Must not register a machine"))
    monkeypatch.setattr(runner.Machine, "load", machine_load)
    monkeypatch.setattr(
        runner.Machine, "get_defaults", lambda: {"machine": "host", "cpu": "test"}
    )
    execute = Mock(return_value=0)
    monkeypatch.setattr(runner, "_run_asv_benchmarks", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark.py",
            "--config",
            str(root / "competition.config.json"),
            "--saps-dir",
            str(tmp_path),
            "--env-dir",
            str(tmp_path / "env"),
            "--results-dir",
            str(tmp_path / "results"),
            "--machine",
            "run_12345-task-0",
            "--chunk-count",
            "5",
            "--chunk-index",
            str(chunk_index),
        ],
    )

    assert runner.main() == 0
    kwargs = execute.call_args.kwargs
    selected = kwargs["benchmarks"]
    actual = [params[index] for index in selected.benchmark_selection.get(name, [])]
    assert (
        actual
        == [
            "jl_projection_inputs.small",
            "jl_projection_inputs.medium",
            "jl_projection_inputs.large",
        ][chunk_index::5]
    )
    assert kwargs["machine_params"].machine == "run_12345-task-0"
    machine_load.assert_not_called()
