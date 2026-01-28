from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import streamlit as st


def _find_pdf_dir() -> Path:
    """Find a folder that contains PDFs.

    Supports both layouts:
    - project_root/app.py + project_root/data/*.pdf
    - project_root/data/app.py + project_root/data/*.pdf
    """

    candidates = [
        Path.cwd() / "data",
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parent,
        Path.cwd(),
    ]

    for c in candidates:
        try:
            if c.exists() and any(
                p.is_file() and p.suffix.lower() == ".pdf" for p in c.iterdir()
            ):
                return c
        except Exception:
            continue

    # Fall back to a sensible default even if empty
    return Path.cwd() / "data"


def _load_llamaindex() -> dict[str, Any]:
    """Import LlamaIndex pieces with compatibility fallbacks."""

    try:
        from llama_index.core import (  # type: ignore
            Settings,
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing LlamaIndex core imports. Install dependencies, e.g.\n"
            "  pip install streamlit llama-index llama-index-readers-file "
            "llama-index-llms-openai llama-index-embeddings-openai pypdf"
        ) from e

    # LLM + embeddings packages are split in newer LlamaIndex versions.
    OpenAI = None
    OpenAIEmbedding = None
    openai_import_error = None

    try:
        from llama_index.llms.openai import OpenAI as _OpenAI  # type: ignore

        OpenAI = _OpenAI
    except Exception as e:  # pragma: no cover
        openai_import_error = e

    try:
        from llama_index.embeddings.openai import (  # type: ignore
            OpenAIEmbedding as _OpenAIEmbedding,
        )

        OpenAIEmbedding = _OpenAIEmbedding
    except Exception as e:  # pragma: no cover
        openai_import_error = openai_import_error or e

    if OpenAI is None or OpenAIEmbedding is None:  # pragma: no cover
        raise ImportError(
            "OpenAI integrations for LlamaIndex are missing. Install, e.g.\n"
            "  pip install llama-index-llms-openai llama-index-embeddings-openai openai"
        ) from openai_import_error

    return {
        "Settings": Settings,
        "SimpleDirectoryReader": SimpleDirectoryReader,
        "StorageContext": StorageContext,
        "VectorStoreIndex": VectorStoreIndex,
        "load_index_from_storage": load_index_from_storage,
        "OpenAI": OpenAI,
        "OpenAIEmbedding": OpenAIEmbedding,
    }


def _extract_sources(resp: Any) -> list[tuple[str, str]]:
    """Return unique (filename, page) tuples from a LlamaIndex response."""

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    source_nodes = getattr(resp, "source_nodes", None) or []
    for nws in source_nodes:
        node = getattr(nws, "node", None) or nws
        meta = getattr(node, "metadata", None) or getattr(nws, "metadata", None) or {}

        file_name = (
            meta.get("file_name")
            or Path(str(meta.get("file_path", ""))).name
            or meta.get("filename")
            or "Unknown file"
        )

        page = (
            meta.get("page_label")
            or meta.get("page_number")
            or meta.get("page")
            or meta.get("page_idx")
        )
        page_str = "?"
        if page is not None and str(page).strip() != "":
            page_str = str(page).strip()

        item = (str(file_name), page_str)
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out


def _sources_html(sources: Iterable[tuple[str, str]]) -> str:
    items = list(sources)
    if not items:
        return ""

    lines = [
        "<div style='color:#6b7280;font-size:0.85rem;line-height:1.35;margin-top:0.5rem;'>",
        "<div style='font-weight:600;'>Sources:</div>",
    ]
    for (fn, pg) in items:
        lines.append(f"<div>• {fn} — p. {pg}</div>")
    lines.append("</div>")
    return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def _get_chat_engine(
    data_dir: str,
    persist_dir: str,
    rebuild_token: int,
    llm_model: str,
    embed_model: str,
):
    li = _load_llamaindex()

    Settings = li["Settings"]
    SimpleDirectoryReader = li["SimpleDirectoryReader"]
    StorageContext = li["StorageContext"]
    VectorStoreIndex = li["VectorStoreIndex"]
    load_index_from_storage = li["load_index_from_storage"]
    OpenAI = li["OpenAI"]
    OpenAIEmbedding = li["OpenAIEmbedding"]

    # Configure default models
    Settings.llm = OpenAI(model=llm_model, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=embed_model)

    persist_path = Path(persist_dir)
    if persist_path.exists() and any(persist_path.iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context)
    else:
        docs = SimpleDirectoryReader(
            input_dir=str(data_dir),
            recursive=False,
            required_exts=[".pdf"],
        ).load_data()
        index = VectorStoreIndex.from_documents(docs)
        persist_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_path))

    # RAG chat engine
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=5,
    )


def main() -> None:
    st.set_page_config(page_title="RAG PDF Assistant", page_icon="📄", layout="centered")

    st.title("RAG PDF Assistant")
    st.caption("Ask questions about the PDFs in your data folder.")

    data_dir = _find_pdf_dir()
    persist_dir = (Path(__file__).resolve().parent / ".rag_storage").resolve()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rebuild_token" not in st.session_state:
        st.session_state.rebuild_token = 0

    # Sidebar controls
    with st.sidebar:
        st.subheader("Data")
        st.write(f"Using: `{data_dir}`")
        pdfs = sorted([p.name for p in data_dir.glob("*.pdf")])
        if pdfs:
            st.write("PDFs:")
            for p in pdfs:
                st.write(f"- {p}")
        else:
            st.warning("No PDFs found in the detected folder.")

        st.subheader("Model")
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="Required to run the default LlamaIndex OpenAI LLM + embeddings.",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        llm_model = st.text_input("LLM model", value="gpt-4o-mini")
        embed_model = st.text_input("Embedding model", value="text-embedding-3-small")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Rebuild index"):
                if persist_dir.exists():
                    shutil.rmtree(persist_dir, ignore_errors=True)
                st.session_state.rebuild_token += 1
                st.rerun()

    if not pdfs:
        st.info("Add PDFs to the data folder, then refresh.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        st.info("Add your OpenAI API key in the sidebar to start chatting.")
        return

    # Build/load chat engine
    try:
        chat_engine = _get_chat_engine(
            data_dir=str(data_dir),
            persist_dir=str(persist_dir),
            rebuild_token=int(st.session_state.rebuild_token),
            llm_model=llm_model,
            embed_model=embed_model,
        )
    except Exception as e:
        st.error(str(e))
        return

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("sources_html"):
                st.markdown(m["sources_html"], unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask a question about the PDFs…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = chat_engine.chat(prompt)
            answer = str(resp)
            sources = _extract_sources(resp)
            sources_block = _sources_html(sources)

            st.markdown(answer)
            if sources_block:
                st.markdown(sources_block, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources_html": sources_block}
    )


if __name__ == "__main__":
    main()
