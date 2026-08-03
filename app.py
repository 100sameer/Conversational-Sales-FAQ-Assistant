"""
Improved Streamlit app for Conversational Sales FAQ Assistant.

Notes:
- This file is defensive about import paths (tries common langchain variants).
- It persists FAISS to "faiss_store" to avoid re-building embeddings every run.
- Uploaded PDFs are written to a secure temporary file and removed after use.
- Retriever and LLM calls are wrapped with fallbacks and error messages.
- Enhanced with modern UI, chat statistics, feedback system, and more features.
"""

import os
import tempfile
import shutil
import traceback
import json
from datetime import datetime

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

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Chat message styling with animations */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px 0;
    }
    
    /* Status indicators */
    .status-online {
        color: #2ecc71;
        font-weight: bold;
    }
    .status-offline {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
    "🎓 Student Counselor": """
You are Aarav, an Academic & Career Advisor at PragyanAI.

Answer ONLY from the context provided below.

Context:
{context}

If the answer is not available in the context, respond with:
"I couldn't find this information in the uploaded documents. Please check with our team for more details."

Always be empathetic, encouraging, and professional in your responses.
""",
    "🏛️ Institution Advisor": """
You are an Institutional Relations Lead at PragyanAI.

Answer only from the context provided.

Context:
{context}

Focus on partnerships, institutional benefits, and program details.
If information is not found, politely suggest contacting the partnerships team.
""",
    "💼 Placement Lead": """
You are an Enterprise Placement Lead at PragyanAI.

Answer only from context.

Context:
{context}

Emphasize placement statistics, company partnerships, and career outcomes.
If placement details are not in context, mention that specific placement data is available upon request.
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
            vs = FAISS.load_local(FAISS_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
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
        docs.append(Document(page_content="PragyanAI AI Program - Comprehensive educational platform offering AI courses, placement assistance, and institutional partnerships."))

    vs = FAISS.from_documents(docs, embeddings)
    try:
        vs.save_local(FAISS_STORE_DIR)
    except Exception:
        st.warning("Could not save FAISS store to disk (permissions?). Continuing without persistence.")
    return vs

vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Settings")

# System Status
st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Status")
status_col1, status_col2 = st.sidebar.columns(2)
llm_status = "🟢 Online" if llm else "🔴 Offline"
db_status = "🟢 Connected" if vectorstore else "🔴 Disconnected"
status_col1.metric("LLM", llm_status)
status_col2.metric("Database", db_status)

st.sidebar.markdown("---")

# Persona Selection
persona = st.sidebar.selectbox("🎯 Choose Persona", list(SALES_PROMPTS.keys()))

st.sidebar.markdown("---")

# Document Upload
uploaded_files = st.sidebar.file_uploader(
    "📄 Upload PDFs",
    accept_multiple_files=True,
    type=["pdf"],
    help="Upload PDF documents to enhance the knowledge base"
)

# Document Count
try:
    doc_count = len(vectorstore.index_to_docstore_id) if hasattr(vectorstore, 'index_to_docstore_id') else 0
    st.sidebar.metric("📚 Documents in DB", doc_count)
except:
    pass

st.sidebar.markdown("---")

# Quick Actions
st.sidebar.subheader("⚡ Quick Actions")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_count = 0
        st.rerun()
with col2:
    if st.button("💾 Export Chat"):
        if st.session_state.messages:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "persona": persona
            }
            json_data = json.dumps(export_data, indent=2)
            st.sidebar.download_button(
                label="📥 Download",
                data=json_data,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.sidebar.warning("No chat history")

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
    progress_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    
    try:
        for i, uploaded in enumerate(uploaded_files):
            progress_text.text(f"📄 Processing: {uploaded.name}")
            tmp_path = safe_write_uploaded_file(uploaded)
            tmp_files.append(tmp_path)
            try:
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()
                docs_to_add.extend(loaded)
            except Exception as e:
                st.sidebar.error(f"Failed to load {uploaded.name}: {e}")
            progress_bar.progress((i + 1) / len(uploaded_files))
        
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
                st.sidebar.success(f"✅ Added {len(docs_to_add)} documents successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to add documents to vectorstore: {e}")
    finally:
        progress_text.empty()
        progress_bar.empty()
        # Cleanup temporary files
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass

# -------------------------------
# MAIN CONTENT
# -------------------------------
# Welcome Banner
st.markdown("""
<div class="welcome-banner">
    <h1>🤖 PragyanAI Assistant</h1>
    <p>Your intelligent companion for academic and career guidance</p>
</div>
""", unsafe_allow_html=True)

# Chat Statistics
if st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💬 Total Messages", len(st.session_state.messages))
    with col2:
        user_msgs = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
        st.metric("👤 User Questions", user_msgs)
    with col3:
        assistant_msgs = len(st.session_state.messages) - user_msgs
        st.metric("🤖 AI Responses", assistant_msgs)

# CHAT HISTORY
st.markdown("---")

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Add feedback buttons for assistant messages
        if msg["role"] == "assistant" and idx > 0:
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"like_{idx}"):
                    st.session_state.feedback[idx] = "positive"
                    st.toast("Thanks for your feedback! 👍", icon="👍")
            with col2:
                if st.button("👎", key=f"dislike_{idx}"):
                    st.session_state.feedback[idx] = "negative"
                    st.toast("We'll improve our responses! 👎", icon="👎")

prompt = st.chat_input("Ask anything about PragyanAI programs...")

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
        
        # Show source documents
        if docs:
            with st.expander("📚 View Source Documents"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_count += 1

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 10px;'>
        <p>🤖 PragyanAI Assistant v2.0 | Powered by <a href='https://groq.com' target='_blank'>Groq</a></p>
        <p style='font-size: 12px;'>Active Persona: {persona} | Model: {GROQ_MODEL}</p>
    </div>
    """,
    unsafe_allow_html=True
)
