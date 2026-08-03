"""
Improved Streamlit app for Conversational Sales FAQ Assistant.

Notes:
- This file is defensive about import paths (tries common langchain variants).
- It persists FAISS to "faiss_store" to avoid re-building embeddings every run.
- Uploaded PDFs are written to a secure temporary file and removed after use.
- Retriever and LLM calls are wrapped with fallbacks and error messages.
"""

import os
import tempfile
import shutil
import traceback

import streamlit as st
import pandas as pd

# Try multiple import paths for compatibility across langchain versions
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    from langchain.document_loaders import PyPDFLoader

try:
    # community name used in your original file
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS

# Embeddings: try langchain_huggingface or langchain built-in wrapper
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings

# Document model
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

# Text splitter for chunking documents
try:
    from langchain.text_splitter import CharacterTextSplitter
except Exception:
    # If not available, we'll fall back to naive chunking
    CharacterTextSplitter = None

# LLM client
from langchain_groq import ChatGroq

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide",
)

# -------------------------------
# CONFIG / SECRETS
# -------------------------------
# Use st.secrets if available, otherwise fall back to environment variable.
GROQ_API_KEY = None
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY is not set. Set it in Streamlit secrets or as env var GROQ_API_KEY.")

# Expose model name as a constant for easier change later
GROQ_MODEL = "llama-3.3-70b-versatile"

# -------------------------------
# LLM INITIALIZATION
# -------------------------------
# Wrap LLM creation so we can show a clear error if the key/model are missing
try:
    llm = ChatGroq(
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
    )
except Exception as e:
    st.error(f"Failed to initialize ChatGroq LLM: {e}")
    llm = None

# -------------------------------
# PERSONAS / SYSTEM PROMPTS
# -------------------------------
SALES_PROMPTS = {
    "PragyanAI Student Counselor": """
You are Aarav, an Academic & Career Advisor.

Answer ONLY from the context.

Context:
{context}

If answer is not available,
say:
'I couldn't find this information in the uploaded documents.'
""",
    "Institution Advisor": """
You are an Institutional Relations Lead.

Answer only from the context.

Context:
{context}
""",
    "Placement Lead": """
You are an Enterprise Placement Lead.

Answer only from context.

Context:
{context}
""",
}

# -------------------------------
# EMBEDDINGS
# -------------------------------
# Use a stable HF sentence-transformers model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# -------------------------------
# VECTORSTORE / DOCUMENT LOADING
# -------------------------------
FAISS_STORE_DIR = "faiss_store"

@st.cache_resource
def build_or_load_vectorstore():
    """
    Build a FAISS index from an Excel fallback or load a persisted index from disk.
    This function is cached by Streamlit so it only runs once per process lifetime.
    """
    # If a persisted index exists, load it
    if os.path.exists(FAISS_STORE_DIR):
        try:
            vs = FAISS.load_local(FAISS_STORE_DIR, embeddings)
            return vs
        except Exception as e:
            st.warning(f"Failed to load existing FAISS store: {e}. Rebuilding...")

    # Otherwise build from an optional Excel fallback file (as in original)
    docs = []
    if os.path.exists("pragyan_faq_prices.xlsx"):
        try:
            df = pd.read_excel("pragyan_faq_prices.xlsx")
            for _, row in df.iterrows():
                text = "\n".join([f"{c}: {row[c]}" for c in df.columns])
                docs.append(Document(page_content=text))
        except Exception as e:
            st.warning(f"Failed to read pragyan_faq_prices.xlsx: {e}")

    if len(docs) == 0:
        docs.append(Document(page_content="PragyanAI AI Program."))

    vs = FAISS.from_documents(docs, embeddings)
    try:
        vs.save_local(FAISS_STORE_DIR)
    except Exception:
        st.warning("Could not save FAISS store to disk (permissions?). Continuing without persistence.")
    return vs

vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Settings")

persona = st.sidebar.selectbox("Choose Persona", list(SALES_PROMPTS.keys()))

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    accept_multiple_files=True,
    type=["pdf"],
)

# -------------------------------
# HELPERS
# -------------------------------
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split long documents into chunks for better retrieval."""
    if CharacterTextSplitter is None:
        # fallback simple chunker
        out = []
        for d in docs:
            text = d.page_content
            for i in range(0, len(text), chunk_size - chunk_overlap):
                out.append(Document(page_content=text[i : i + chunk_size]))
        return out

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    out = []
    for d in docs:
        texts = splitter.split_text(d.page_content)
        out.extend([Document(page_content=t) for t in texts])
    return out

def safe_write_uploaded_file(uploaded_file) -> str:
    """
    Write a Streamlit UploadedFile to a secure temporary file and return path.
    Caller should remove the file after use.
    """
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

def query_llm_with_fallback(llm_client, prompt_text):
    """
    Call the LLM with several fallbacks depending on the client API surface.
    Return a string answer or raise.
    """
    if llm_client is None:
        raise RuntimeError("LLM client is not initialized.")
    last_exc = None
    # Try common call patterns
    try:
        # your original style (ChatGroq.invoke)
        resp = llm_client.invoke(prompt_text)
        # If response has attribute 'content'
        if hasattr(resp, "content"):
            return resp.content
        # If it's a plain string
        if isinstance(resp, str):
            return resp
        return str(resp)
    except Exception as e:
        last_exc = e

    try:
        # try __call__ style
        resp = llm_client(prompt_text)
        if isinstance(resp, str):
            return resp
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        last_exc = e

    # If everything failed, raise the last exception with a helpful message
    raise RuntimeError(f"LLM call failed. Last error: {last_exc}")

# -------------------------------
# ADD NEW PDFS (UPLOAD)
# -------------------------------
if uploaded_files:
    docs_to_add = []
    tmp_files = []
    try:
        for uploaded in uploaded_files:
            tmp_path = safe_write_uploaded_file(uploaded)
            tmp_files.append(tmp_path)
            try:
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()
                docs_to_add.extend(loaded)
            except Exception as e:
                st.sidebar.error(f"Failed to load {uploaded.name}: {e}")
        if docs_to_add:
            # chunk before adding
            chunks = chunk_documents(docs_to_add)
            try:
                vectorstore.add_documents(chunks)
                # Try to persist the updated index
                try:
                    vectorstore.save_local(FAISS_STORE_DIR)
                except Exception:
                    st.sidebar.warning("Could not persist updated FAISS index.")
                st.sidebar.success("Documents added to vectorstore!")
            except Exception as e:
                st.sidebar.error(f"Failed to add documents to vectorstore: {e}")
    finally:
        # Cleanup temporary files
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass

# -------------------------------
# CHAT HISTORY
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 PragyanAI AI Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant docs
    try:
        docs = retriever.get_relevant_documents(prompt)
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        docs = []

    context = "\n\n".join(d.page_content for d in docs)

    system_prompt = SALES_PROMPTS[persona].format(context=context)
    full_prompt = system_prompt + "\n\nUser: " + prompt

    try:
        answer = query_llm_with_fallback(llm, full_prompt)
    except Exception as e:
        st.error("LLM call failed: " + str(e))
        # for debugging (optional, remove in production)
        st.debug(traceback.format_exc())
        answer = "I couldn't process the request at this time."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
