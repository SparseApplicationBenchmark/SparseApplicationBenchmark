from __future__ import annotations

import numpy as np
import pytest

from scipy import sparse as scipy_sparse

from binsparse.conversions import to_numpy, to_scipy

from saps.benchmarks import ode
from saps.benchmarks.ode import (
    SLICOTDataset,
    SLICOTForwardEuler,
    SLICOTGenerator,
    SLICOTRK4,
)


def _source_meta(source_name: str) -> dict:
    return {
        "source_url": f"https://example.test/{source_name}.zip",
        "source_page_url": "https://example.test/slicot",
        "local_path": f"/tmp/{source_name}",
    }


def test_slicot_generator_uses_all_identity_e_problems():
    generator = SLICOTGenerator()

    assert [dataset.source_name for dataset in generator.datasets] == [
        "eady.mat",
        "CDplayer.mat",
        "fom.mat",
        "random.mat",
        "pde.mat",
        "heat-cont.mat",
        "Orr-Som.mat",
        "iss.mat",
        "build.mat",
        "beam.mat",
    ]
    assert all(dataset.suites == ["standard"] for dataset in generator.datasets)


def test_slicot_generator_loads_a_and_b_and_ignores_c_d(monkeypatch):
    def fake_load_slicot_problem(source_name):
        return (
            {
                "A": np.array([[0.0, 1.0], [-2.0, -3.0]]),
                "B": np.array([[0.0], [1.0]]),
                "C": np.array([[1.0, 0.0]]),
                "D": np.array([[0.0]]),
            },
            _source_meta(source_name),
        )

    monkeypatch.setattr(ode, "load_slicot_problem", fake_load_slicot_problem)

    instance = SLICOTGenerator().generate(SLICOTDataset("build"))

    np.testing.assert_array_equal(
        to_numpy(instance.inputs[0]), np.array([[0.0, 1.0], [-2.0, -3.0]])
    )
    np.testing.assert_array_equal(to_numpy(instance.inputs[1]), np.array([[0.0], [1.0]]))
    assert instance.meta["source_name"] == "build.mat"
    assert instance.meta["assumed_E"] == "identity"
    assert instance.meta["assumed_B"] is None
    assert instance.meta["A_storage"] == "dense"
    assert instance.meta["B_storage"] == "dense"
    assert "C" not in instance.meta
    assert "D" not in instance.meta


def test_slicot_generator_preserves_stored_sparse_matrices(monkeypatch):
    A = scipy_sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    B = scipy_sparse.csc_matrix(np.array([[0.0], [3.0]]))

    def fake_load_slicot_problem(source_name):
        return ({"A": A, "B": B}, _source_meta(source_name))

    monkeypatch.setattr(ode, "load_slicot_problem", fake_load_slicot_problem)

    instance = SLICOTGenerator().generate(SLICOTDataset("heat-cont"))

    assert instance.meta["A_storage"] == "sparse"
    assert instance.meta["B_storage"] == "sparse"
    assert scipy_sparse.issparse(to_scipy(instance.inputs[0]))
    assert scipy_sparse.issparse(to_scipy(instance.inputs[1]))
    np.testing.assert_array_equal(to_scipy(instance.inputs[0]).toarray(), A.toarray())
    np.testing.assert_array_equal(to_scipy(instance.inputs[1]).toarray(), B.toarray())
    np.testing.assert_array_equal(ode._dense_binsparse_array(instance.inputs[0]), A.toarray())


def test_slicot_generator_keeps_stored_dense_matrices_dense(monkeypatch):
    def fake_load_slicot_problem(source_name):
        return (
            {
                "A": np.array([[1.0, 0.0], [0.0, 2.0]]),
                "B": np.array([[0.0], [3.0]]),
            },
            _source_meta(source_name),
        )

    monkeypatch.setattr(ode, "load_slicot_problem", fake_load_slicot_problem)

    instance = SLICOTGenerator().generate(SLICOTDataset("beam"))

    assert instance.meta["A_storage"] == "dense"
    assert instance.meta["B_storage"] == "dense"
    with pytest.raises(TypeError):
        to_scipy(instance.inputs[0])
    with pytest.raises(TypeError):
        to_scipy(instance.inputs[1])


def test_slicot_generator_defaults_missing_b_to_normalized_input(monkeypatch):
    def fake_load_slicot_problem(source_name):
        return (
            {"A": np.array([[1.0, 2.0], [3.0, 4.0]])},
            _source_meta(source_name),
        )

    monkeypatch.setattr(ode, "load_slicot_problem", fake_load_slicot_problem)

    instance = SLICOTGenerator().generate(SLICOTDataset("Orr-Som"))

    np.testing.assert_allclose(to_numpy(instance.inputs[1]), np.ones((2, 1)) / np.sqrt(2))
    assert instance.meta["source_name"] == "Orr-Som.mat"
    assert instance.meta["assumed_B"] == "normalized_uniform_vector"
    assert instance.meta["source_inputs"] == 1
    assert instance.meta["input_dimension"] == 1
    assert instance.meta["B_storage"] == "dense"


def test_slicot_generator_rejects_explicit_e(monkeypatch):
    def fake_load_slicot_problem(source_name):
        return (
            {
                "A": np.eye(2),
                "B": np.ones((2, 1)),
                "E": np.eye(2),
            },
            _source_meta(source_name),
        )

    monkeypatch.setattr(ode, "load_slicot_problem", fake_load_slicot_problem)

    with pytest.raises(ValueError, match="explicit E matrix"):
        SLICOTGenerator().generate(SLICOTDataset("eady"))


def test_slicot_forward_euler_runs_linear_system():
    benchmark = SLICOTForwardEuler()
    data = [np.array([[0.0]]), np.array([[2.0]])]
    meta = {
        "span": (0.0, 0.3),
        "y0": [0.0],
        "step": 0.1,
        "input_value": 3.0,
    }

    time, states = benchmark.benchmark(None, data, meta)

    np.testing.assert_allclose(time, np.array([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(states[:, 0], np.array([0.0, 0.6, 1.2]))


def test_slicot_rk4_is_parented_benchmark():
    generator_names = [generator.name for generator in SLICOTRK4().generators]

    assert generator_names == ["slicot_ode"]
