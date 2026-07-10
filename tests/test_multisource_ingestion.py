from __future__ import annotations

import zipfile
from pathlib import Path

from ragbase.ingestor import (
    load_code_dir_documents,
    load_mixed_documents,
    load_path_documents,
    load_url_documents,
)


def test_app_keeps_current_project_source_channels():
    app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    for label in ("文件", "URL/GitHub", "代码目录", "混合导入"):
        assert label in app_text
    for ext in ("pdf", "docx", "md", "txt"):
        assert ext in app_text

    assert "Wiki" not in app_text
    assert "load_wiki_documents" not in app_text


def test_loads_text_markdown_docx_and_code_dir(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("plain text source", encoding="utf-8")
    md = tmp_path / "guide.md"
    md.write_text("# guide\nmarkdown source", encoding="utf-8")
    docx = tmp_path / "outline.docx"
    _write_minimal_docx(docx, "docx source text")

    code_dir = tmp_path / "repo"
    code_dir.mkdir()
    (code_dir / "main.py").write_text("print('code source')", encoding="utf-8")
    (code_dir / "ignore.exe").write_text("ignore", encoding="utf-8")

    docs = []
    docs.extend(load_path_documents(txt))
    docs.extend(load_path_documents(md))
    docs.extend(load_path_documents(docx))
    docs.extend(load_code_dir_documents(code_dir))

    joined = "\n".join(doc.page_content for doc in docs)
    assert "plain text source" in joined
    assert "markdown source" in joined
    assert "docx source text" in joined
    assert "code source" in joined
    assert all(doc.metadata.get("source_name") for doc in docs)


def test_loads_url_and_mixed_sources(tmp_path, monkeypatch):
    txt = tmp_path / "mixed.txt"
    txt.write_text("mixed local source", encoding="utf-8")

    class Response:
        status_code = 200
        text = "<html><body><h1>remote source</h1></body></html>"

        def raise_for_status(self):
            return None

    def fake_get(url, timeout=20, headers=None):
        return Response()

    monkeypatch.setattr("requests.get", fake_get)

    url_docs = load_url_documents("https://example.com/article")
    mixed_docs = load_mixed_documents([str(txt), "https://example.com/article"])

    assert "remote source" in url_docs[0].page_content
    assert "mixed local source" in "\n".join(doc.page_content for doc in mixed_docs)
    assert "remote source" in "\n".join(doc.page_content for doc in mixed_docs)


def _write_minimal_docx(path: Path, text: str) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>"
        f"{text}"
        "</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
