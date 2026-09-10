from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import numpy as np
import scipy.sparse as sp

import boto3
import h5py
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy
from botocore.exceptions import ClientError
from filelock import FileLock

import saps.storage
from saps.benchmark import DataInstance
from saps.storage import LocalStorageBackend, S3StorageBackend


def _expired_token_error(operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "ExpiredToken",
                "Message": "The provided token has expired",
            }
        },
        operation,
    )


class _FakeS3Client:
    def __init__(
        self,
        name: str,
        calls: list[tuple[str, str]],
        *,
        fail_head: bool = False,
        fail_download: bool = False,
    ) -> None:
        self.name = name
        self.calls = calls
        self.fail_head = fail_head
        self.fail_download = fail_download

    def head_object(self, *, Bucket: str, Key: str) -> None:
        self.calls.append((self.name, f"head:{Bucket}/{Key}"))
        if self.fail_head:
            raise _expired_token_error("HeadObject")

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.calls.append((self.name, f"download:{bucket}/{key}"))
        if self.fail_download:
            raise _expired_token_error("GetObject")
        Path(filename).write_text("{}", encoding="utf-8")


def _backend_with_fake_clients(monkeypatch, tmp_path, signed, unsigned):
    def fake_client(service_name, *, config=None):
        assert service_name == "s3"
        return unsigned if config is not None else signed

    monkeypatch.setattr(boto3, "client", fake_client)
    return S3StorageBackend("s3://example-bucket", tmp_path / "manifest.json", tmp_path)


def test_s3_file_exists_checks_public_object_without_signed_credentials(
    monkeypatch, tmp_path
):
    calls: list[tuple[str, str]] = []
    signed = _FakeS3Client("signed", calls, fail_head=True)
    unsigned = _FakeS3Client("unsigned", calls)
    backend = _backend_with_fake_clients(monkeypatch, tmp_path, signed, unsigned)

    assert backend.file_exists("datasets/example.json")
    assert calls == [("unsigned", "head:example-bucket/datasets/example.json")]


def test_s3_download_reads_public_object_without_signed_credentials(
    monkeypatch, tmp_path
):
    calls: list[tuple[str, str]] = []
    signed = _FakeS3Client("signed", calls, fail_download=True)
    unsigned = _FakeS3Client("unsigned", calls)
    backend = _backend_with_fake_clients(monkeypatch, tmp_path, signed, unsigned)
    local_path = tmp_path / "downloaded.json"

    assert backend.download_file("datasets/example.json", local_path)
    assert local_path.read_text(encoding="utf-8") == "{}"
    assert calls == [("unsigned", "download:example-bucket/datasets/example.json")]


def test_data_round_trips_through_binsparse_hdf5(tmp_path):
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    dense = np.arange(6, dtype=np.float32).reshape(2, 3)
    sparse = sp.coo_array(
        (np.array([1.5, 2.5]), (np.array([0, 2]), np.array([1, 0]))),
        shape=(3, 2),
    )
    data = DataInstance(
        inputs=[from_numpy(dense), from_scipy(sparse)],
        meta={"source": "test"},
        ref_outputs=[from_numpy(dense + 1)],
        ref_meta={"tolerance": 1e-6},
    )
    path = tmp_path / "dataset.bsp.h5"

    digest = backend.serialize_data_to_file(data, path)
    restored = backend.deserialize_data_from_file(path)

    with h5py.File(path, "r") as file:
        assert set(file) == {"inputs", "ref_outputs"}
        assert set(file["inputs"]) == {"0", "1"}
        descriptor = json.loads(file["inputs/0"].attrs["binsparse"])
        assert descriptor["binsparse"]["format"] == "DMATR"
    assert np.array_equal(to_numpy(restored.inputs[0]), dense)
    assert np.array_equal(to_scipy(restored.inputs[1]).toarray(), sparse.toarray())
    assert restored.meta == data.meta
    assert restored.ref_meta == data.ref_meta
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def downloadable_dataset(tmp_path):
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    generator = SimpleNamespace(
        name="example",
        generate=Mock(side_effect=RuntimeError("Unexpected generation")),
    )
    dataset = SimpleNamespace(name="small", file="example.py", freshness="v1")
    data = DataInstance(inputs=[from_numpy(np.arange(6))], meta={"source": "test"})
    source = tmp_path / "source.bsp.h5"
    digest = backend.serialize_data_to_file(data, source)
    prefix = backend.prefix(generator, dataset, digest)
    assert backend.upload_file(source, prefix)
    backend.update_manifest(generator, dataset, digest)
    return backend, generator, dataset, backend.cache_dir / prefix


def test_concurrent_requests_download_once_and_reuse_shared_cache(
    downloadable_dataset, monkeypatch
):
    backend, generator, dataset, cache_path = downloadable_dataset
    download = backend.download_file
    waiter_blocked = Event()

    class ObservedFileLock(FileLock):
        def _acquire(self):
            super()._acquire()
            if not self.is_locked:
                waiter_blocked.set()

    monkeypatch.setattr(saps.storage, "FileLock", ObservedFileLock)

    def partial_download(prefix, path):
        path.write_bytes(b"partial download")
        assert not cache_path.exists()
        assert waiter_blocked.wait(timeout=10)
        return download(prefix, path)

    download_calls = Mock(side_effect=partial_download)
    monkeypatch.setattr(backend, "download_file", download_calls)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(backend.retrieve_dataset, generator, dataset)
            for _ in range(2)
        ]
        for future in futures:
            assert np.array_equal(to_numpy(future.result().inputs[0]), np.arange(6))
    assert download_calls.call_count == 1

    next_run = LocalStorageBackend(
        backend.base_path, backend.manifest_path, backend.cache_dir
    )
    monkeypatch.setattr(
        next_run, "download_file", Mock(side_effect=AssertionError("Already cached"))
    )
    assert np.array_equal(
        to_numpy(next_run.retrieve_dataset(generator, dataset).inputs[0]), np.arange(6)
    )
    generator.generate.assert_not_called()
    assert not list(backend.cache_dir.rglob(".saps-*"))


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [("checksum", AssertionError), ("interrupted", OSError), ("failed", RuntimeError)],
)
def test_failed_downloads_do_not_poison_shared_cache(
    downloadable_dataset, monkeypatch, failure, expected_error
):
    backend, generator, dataset, cache_path = downloadable_dataset
    download = backend.download_file

    def broken_download(prefix, path):
        path.write_bytes(b"partial download")
        if failure == "interrupted":
            raise OSError("Download interrupted")
        return failure != "failed"

    monkeypatch.setattr(backend, "download_file", broken_download)
    with pytest.raises(expected_error):
        backend.retrieve_dataset(generator, dataset)
    assert not cache_path.exists()
    assert not list(backend.cache_dir.rglob(".saps-*"))

    monkeypatch.setattr(backend, "download_file", download)
    assert np.array_equal(
        to_numpy(backend.retrieve_dataset(generator, dataset).inputs[0]), np.arange(6)
    )
