from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from jfk_reader.core import chunks, build_index, retrieve, answer


def test_chunks_are_bounded_and_overlap():
    parts = chunks("a " * 500, size=100, overlap=20)
    assert len(parts) > 1
    assert all(len(p) <= 100 for p in parts)
    assert "" not in parts


def test_chunks_empty_and_bad_args():
    assert chunks("  ") == []
    with pytest.raises(ValueError):
        chunks("abc", 10, 10)


class FakeEmbeddings:
    def create(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, 0.0] if "Kennedy" in s else [0.0, 1.0]) for i, s in enumerate(input)])


class FakeClient:
    embeddings = FakeEmbeddings()
    chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="The excerpt mentions Kennedy [S1]."))])))


def test_index_and_ask_without_network(tmp_path, monkeypatch):
    from jfk_reader import core
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "record.pdf").write_bytes(b"test placeholder")
    monkeypatch.setattr(core, "extract_pages", lambda path, use_ocr=False: [{"page": 3, "text": "Kennedy records describe a document."}])
    index = tmp_path / "index"
    assert build_index(pdfs, index, FakeClient()) == 1
    hits = retrieve("Kennedy", index, FakeClient())
    assert hits[0]["page"] == 3
    response, sources = answer("Kennedy?", index, FakeClient())
    assert "[S1]" in response
    assert sources[0]["file"] == "record.pdf"
