"""
Improved Streamlit app for Conversational Sales FAQ Assistant.

Main improvements:
- Safer secrets access and configurable model/retrieval settings via sidebar.
- Better error handling, logging, and spinners for long ops.
- Persist and show source metadata for uploaded PDFs.
- More robust LLM call fallbacks and more helpful user-facing errors.
- Chunking parameters configurable and fallback chunker preserved.
"""

from __future__ import annotations

import os
import tempfile
import traceback
import logging
from typing import List, Optional, Dict

import streamlit as st
import pandas as pd

# configure logger
logger = logging.getLogger("pragyanai_assistant")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -------------------------------
# Flexible imports for LangChain variants
# -------------------------------
def try_imports():
    # document loader
    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
    except Exception:
        from langchain.document_loaders import PyPDFLoader  # type: ignore

    # FAISS vectorstore
    try:
        from langchain_community.vectorstores import FAISS  # type: ignore
    except Exception:
        from langchain.vectorstores import FAISS  # type: ignore

    # Embeddings wrappers
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

    # Document model
    try:
        from langchain_core.documents import Document  # type: ignore
    except Exception:
        from langchain.schema import Document  # type: ignore

    # Text splitter
    try:
        from langchain.text_splitter import CharacterTextSplitter  # type: ignore
    except Exception:
        CharacterTextSplitter = None  # type: ignore

    # LLM client (ChatGroq wrapper)
    try:
        from langchain_groq import ChatGroq  # type: ignore
    except Exception:
        ChatGroq = None  # type: ignore

    return {
        "PyPDFLoader": PyPDFLoader,
        "FAISS": FAISS,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        "Document": Document,
        "CharacterTextSplitter": CharacterTextSplitter,
        "ChatGroq": ChatGroq,
    }


_imports = try_imports()
PyPDFLoader = _imports["PyPDFLoader"]
FAISS = _imports["FAISS"]
HuggingFaceEmbeddings = _imports["HuggingFaceEmbeddings"]
Document = _imports["Document"]
CharacterTextSplitter = _imports["CharacterTextSplitter"]
ChatGroq = _imports["ChatGroq"]

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="PragyanAI Assistant", page_icon="🤖", layout="wide")

# -------------------------------
# Config / Secrets
# -------------------------------
# Prefer st.secrets, fallback to env var.
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
GROQ_API_KEY = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY is not set. Set it in Streamlit secrets or as env var GROQ_API_KEY.")
    logger.warning("GROQ_API_KEY not found in st.secrets or environment variables.")

# Defaults and sidebar-configurable options
DEFAULT_MODEL = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile") if hasattr(st, "secrets") else "llama-3.3-70b-versatile"
DEFAULT_EMBED_MODEL = st.secrets.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") if hasattr(st, "secrets") else "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.title("Settings")
model_name = st.sidebar.text_input("LLM model", DEFAULT_MODEL)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
retrieval_k = st.sidebar.number_input("Retriever k (how many docs to fetch)", min_value=1, max_value=10, value=4, step=1)
chunk_size = st.sidebar.number_input("Chunk size (characters)", min_value=256, max_value=4000, value=1000, step=128)
chunk_overlap = st.sidebar.number_input("Chunk overlap (characters)", min_value=0, max_value=1024, value=200, step=32)
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
persona = st.sidebar.selectbox("Choose Persona", ["PragyanAI Student Counselor", "Institution Advisor", "Placement Lead"])

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# -------------------------------
# Personas / system prompts
# -------------------------------
SALES_PROMPTS = {
    "PragyanAI Student Counselor": """You are Aarav, an Academic & Career Advisor.

Answer ONLY from the context.

Context:
{context}

If answer is not available,
say:
'I couldn't find this information in the uploaded documents.'""",
    "Institution Advisor": """You are an Institutional Relations Lead.

Answer only from the context.

Context:
{context}""",
    "Placement Lead": """You are an Enterprise Placement Lead.

Answer only from context.

Context:
{context}""",
}

# -------------------------------
# Embeddings and vectorstore (cached)
# -------------------------------
EMBEDDING_MODEL = DEFAULT_EMBED_MODEL
FAISS_STORE_DIR = "faiss_store"

@st.cache_resource
def get_embeddings():
    try:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return emb
    except Exception as e:
        logger.exception("Failed to initialize embeddings: %s", e)
        raise

@st.cache_resource
def build_or_load_vectorstore(embeddings):
    """
    Build or load a FAISS index. This runs once per Streamlit process lifetime.
    """
    # Try to load
    if os.path.exists(FAISS_STORE_DIR):
        try:
            vs = FAISS.load_local(FAISS_STORE_DIR, embeddings)
            logger.info("Loaded FAISS store from %s", FAISS_STORE_DIR)
            return vs
        except Exception as e:
            logger.warning("Failed to load existing FAISS store: %s. Rebuilding...", e)

    docs = []
    if os.path.exists("pragyan_faq_prices.xlsx"):
        try:
            df = pd.read_excel("pragyan_faq_prices.xlsx")
            for _, row in df.iterrows():
                text = "\n".join([f"{c}: {row[c]}" for c in df.columns])
                docs.append(Document(page_content=text, metadata={"source": "pragyan_faq_prices.xlsx"}))
        except Exception as e:
            logger.warning("Failed to read pragyan_faq_prices.xlsx: %s", e)

    if not docs:
        docs.append(Document(page_content="PragyanAI AI Program.", metadata={"source": "fallback"}))

    try:
        vs = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        logger.exception("Failed to build FAISS from documents: %s", e)
        raise

    try:
        vs.save_local(FAISS_STORE_DIR)
    except Exception:
        logger.warning("Could not save FAISS store to disk (permissions?). Continuing without persistence.")
    return vs

embeddings = get_embeddings()
vectorstore = build_or_load_vectorstore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

# Show document count safely
try:
    doc_count = getattr(vectorstore, "index_to_docstore_id", None)
    if doc_count is not None:
        doc_count = len(doc_count)
    else:
        # fallback: try to read docstore if available
        doc_count = len(list(vectorstore._docstore._dict.keys())) if hasattr(vectorstore, "_docstore") else "unknown"
    st.sidebar.write(f"📚 Documents: {doc_count}")
except Exception:
    logger.exception("Failed to determine document count for sidebar.")

# -------------------------------
# LLM initialization
# -------------------------------
llm = None
if ChatGroq is not None:
    try:
        with st.spinner("Initializing LLM..."):
            llm = ChatGroq(model_name=model_name, groq_api_key=GROQ_API_KEY, temperature=temperature)
    except Exception as e:
        logger.exception("Failed to initialize ChatGroq: %s", e)
        st.error(f"Failed to initialize LLM: {e}")
else:
    logger.warning("ChatGroq client not found. LLM features will be disabled.")
    st.error("ChatGroq client not available in environment. Include langchain_groq or compatible client.")

# -------------------------------
# Helpers
# -------------------------------
def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    if CharacterTextSplitter is None:
        out = []
        for d in docs:
            text = d.page_content
            for i in range(0, len(text), chunk_size - chunk_overlap):
                out.append(Document(page_content=text[i : i + chunk_size], metadata=d.metadata or {}))
        return out

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for d in docs:
        texts = splitter.split_text(d.page_content)
        out.extend([Document(page_content=t, metadata=d.metadata or {}) for t in texts])
    return out

def safe_write_uploaded_file(uploaded_file) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except Exception:
            pass
        raise

def extract_answer_from_resp(resp) -> str:
    # Normalize different LLM return shapes into a string
    try:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        # Common structures
        if hasattr(resp, "content"):
            return getattr(resp, "content")
        if isinstance(resp, dict):
            # e.g., {"choices":[{"text": "..."}]}
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                return first.get("text") or first.get("message") or str(first)
            return str(resp)
        # fallback to str()
        return str(resp)
    except Exception as e:
        logger.exception("Failed to extract answer: %s", e)
        return str(resp)

def query_llm_with_fallback(llm_client, prompt_text: str, timeout: int = 60) -> str:
    if llm_client is None:
        raise RuntimeError("LLM client is not initialized.")
    last_exc = None
    # Common call patterns
    try:
        resp = None
        # Try invoke
        try:
            resp = llm_client.invoke(prompt_text)
        except Exception:
            resp = None

        if resp is None:
            try:
                resp = llm_client(prompt_text)
            except Exception:
                resp = None

        if resp is None:
            # Some clients expect a messages list:
            try:
                resp = llm_client.invoke({"messages": [{"role": "system", "content": ""}, {"role": "user", "content": prompt_text}]})
            except Exception:
                resp = None

        if resp is None:
            raise RuntimeError("No supported call pattern succeeded.")

        return extract_answer_from_resp(resp)
    except Exception as e:
        logger.exception("LLM call failure: %s", e)
        raise RuntimeError(f"LLM call failed: {e}")

# -------------------------------
# Handle uploads (add PDFs)
# -------------------------------
if uploaded_files:
    docs_to_add = []
    tmp_files = []
    try:
        for uploaded in uploaded_files:
            # safety: reject huge files (example limit)
            if uploaded.size and uploaded.size > 30 * 1024 * 1024:
                st.sidebar.error(f"{uploaded.name} is larger than 30MB and was skipped.")
                continue
            tmp_path = safe_write_uploaded_file(uploaded)
            tmp_files.append(tmp_path)
            try:
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()
                # Attach filename metadata to each page/document
                for d in loaded:
                    d.metadata = d.metadata or {}
                    d.metadata["source"] = uploaded.name
                docs_to_add.extend(loaded)
            except Exception as e:
                logger.exception("Failed to load %s: %s", uploaded.name, e)
                st.sidebar.error(f"Failed to load {uploaded.name}: {e}")

        if docs_to_add:
            with st.spinner("Chunking and adding documents to vectorstore..."):
                chunks = chunk_documents(docs_to_add, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                try:
                    vectorstore.add_documents(chunks)
                    try:
                        vectorstore.save_local(FAISS_STORE_DIR)
                    except Exception:
                        logger.warning("Could not persist updated FAISS index.")
                        st.sidebar.warning("Could not persist updated FAISS index.")
                    st.sidebar.success("Documents added to vectorstore!")
                    # reload retriever with the chosen k
                    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
                    st.experimental_rerun()
                except Exception as e:
                    logger.exception("Failed to add documents to vectorstore: %s", e)
                    st.sidebar.error(f"Failed to add documents to vectorstore: {e}")
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass

# -------------------------------
# Chat history state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 PragyanAI AI Assistant")

# chat stats
if st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(st.session_state.messages))
    with col2:
        user_msgs = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
        st.metric("User Questions", user_msgs)
    with col3:
        assistant_msgs = len(st.session_state.messages) - user_msgs
        st.metric("AI Responses", assistant_msgs)
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # retrieval
    try:
        with st.spinner("Retrieving relevant documents..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
            docs = retriever.get_relevant_documents(prompt)
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        st.error(f"Retrieval failed: {e}")
        docs = []

    context = "\n\n".join(d.page_content for d in docs) if docs else ""
    system_prompt = SALES_PROMPTS.get(persona, SALES_PROMPTS["PragyanAI Student Counselor"]).format(context=context)
    full_prompt = system_prompt + "\n\nUser: " + prompt

    try:
        with st.spinner("Contacting LLM..."):
            answer = query_llm_with_fallback(llm, full_prompt)
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        st.error("I couldn't process the request at this time.")
        answer = "I couldn't process the request at this time."

    with st.chat_message("assistant"):
        st.markdown(answer)
        if docs:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(docs, 1):
                    source = (doc.metadata or {}).get("source", "unknown")
                    snippet = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    st.write(f"**Source {i}:** {source}")
                    st.write(snippet)
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
