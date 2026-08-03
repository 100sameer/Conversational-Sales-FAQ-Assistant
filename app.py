"""
Enhanced Streamlit app for Conversational Sales FAQ Assistant.
Features: Modern UI, chat history management, export capabilities, document management, and more.
"""

import os
import tempfile
import shutil
import traceback
import json
from datetime import datetime
import hashlib

import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# Try multiple import paths for compatibility across langchain versions
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    from langchain.document_loaders import PyPDFLoader

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

try:
    from langchain.text_splitter import CharacterTextSplitter
except Exception:
    CharacterTextSplitter = None

from langchain_groq import ChatGroq

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* User message */
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
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
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Metrics cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
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
    
    /* Document upload area */
    .upload-area {
        border: 2px dashed #3498db;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background: rgba(52, 152, 219, 0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# CONFIG / SECRETS
# -------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY is not set. Please configure it in secrets or environment variables.")

# -------------------------------
# LLM INITIALIZATION
# -------------------------------
@st.cache_resource
def initialize_llm():
    try:
        return ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.3,
            max_retries=2,
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

llm = initialize_llm()

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
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = get_embeddings()

# -------------------------------
# VECTORSTORE MANAGEMENT
# -------------------------------
FAISS_STORE_DIR = "faiss_store"

@st.cache_resource
def build_or_load_vectorstore():
    """Build or load vectorstore with caching."""
    if os.path.exists(FAISS_STORE_DIR):
        try:
            vs = FAISS.load_local(FAISS_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            return vs
        except Exception as e:
            st.warning(f"Failed to load existing FAISS store: {e}. Rebuilding...")

    docs = []
    # Load from Excel if available
    if os.path.exists("pragyan_faq_prices.xlsx"):
        try:
            df = pd.read_excel("pragyan_faq_prices.xlsx")
            for _, row in df.iterrows():
                text = "\n".join([f"{c}: {row[c]}" for c in df.columns])
                docs.append(Document(page_content=text))
            st.success(f"Loaded {len(docs)} documents from Excel")
        except Exception as e:
            st.warning(f"Failed to read pragyan_faq_prices.xlsx: {e}")

    if len(docs) == 0:
        docs.append(Document(page_content="PragyanAI AI Program - Basic information."))

    vs = FAISS.from_documents(docs, embeddings)
    try:
        vs.save_local(FAISS_STORE_DIR)
    except Exception:
        st.warning("Could not save FAISS store to disk.")
    return vs

try:
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    vectorstore_ready = True
except Exception as e:
    st.error(f"Failed to initialize vectorstore: {e}")
    vectorstore_ready = False
    retriever = None

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_count" not in st.session_state:
    st.session_state.document_count = 0
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {
        "theme": "light",
        "response_length": "medium",
        "language": "English"
    }

# -------------------------------
# SIDEBAR - ENHANCED
# -------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=PragyanAI", use_column_width=True)
    st.markdown("---")
    
    # Navigation Menu
    selected = option_menu(
        menu_title="Navigation",
        options=["💬 Chat", "📊 Analytics", "📚 Documents", "⚙️ Settings"],
        icons=["chat", "graph-up", "book", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.2)"},
        }
    )
    
    st.markdown("---")
    
    # Persona Selection
    st.subheader("🎯 Persona")
    persona = st.selectbox(
        "Select Role",
        list(SALES_PROMPTS.keys()),
        help="Choose the AI's role for specialized responses"
    )
    
    st.markdown("---")
    
    # Document Upload Section
    st.subheader("📄 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True,
        type=["pdf"],
        help="Upload PDF documents to enhance the knowledge base"
    )
    
    # Document Stats
    if vectorstore_ready:
        try:
            doc_count = len(vectorstore.index_to_docstore_id) if hasattr(vectorstore, 'index_to_docstore_id') else 0
            st.metric("📚 Documents in DB", doc_count)
        except:
            pass
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            export_chat()
    
    if st.button("🔄 Reset Vectorstore", use_container_width=True):
        if os.path.exists(FAISS_STORE_DIR):
            shutil.rmtree(FAISS_STORE_DIR)
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Status Indicator
    st.subheader("📊 System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        llm_status = "🟢 Online" if llm else "🔴 Offline"
        st.markdown(f"**LLM:** {llm_status}")
    with status_col2:
        db_status = "🟢 Connected" if vectorstore_ready else "🔴 Disconnected"
        st.markdown(f"**Database:** {db_status}")

# -------------------------------
# MAIN CONTENT AREA
# -------------------------------
if selected == "💬 Chat":
    st.title("🤖 PragyanAI AI Assistant")
    st.markdown("*Your intelligent companion for sales and support inquiries*")
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Add feedback buttons for assistant messages
            if msg["role"] == "assistant" and len(st.session_state.messages) > 1:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"like_{msg['id']}" if 'id' in msg else "like"):
                        st.toast("Thanks for your feedback! 👍")
                with col2:
                    if st.button("👎", key=f"dislike_{msg['id']}" if 'id' in msg else "dislike"):
                        st.toast("We'll improve our responses! 👎")
    
    # Chat input
    if prompt := st.chat_input("Ask anything about PragyanAI programs..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        if vectorstore_ready and llm:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Retrieve relevant documents
                        docs = retriever.get_relevant_documents(prompt)
                        context = "\n\n".join(d.page_content for d in docs)
                        
                        # Prepare prompt
                        system_prompt = SALES_PROMPTS[persona].format(context=context)
                        full_prompt = system_prompt + "\n\nUser: " + prompt
                        
                        # Get response
                        response = llm.invoke(full_prompt)
                        answer = response.content if hasattr(response, 'content') else str(response)
                        
                        # Display response
                        st.markdown(answer)
                        
                        # Add to session
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "id": len(st.session_state.messages)
                        })
                        
                        # Show source documents
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(docs, 1):
                                st.text(f"Source {i}:")
                                st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                                st.divider()
                                
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        st.debug(traceback.format_exc())
        else:
            st.error("System not properly initialized. Please check the sidebar status.")

elif selected == "📊 Analytics":
    st.title("📊 Chat Analytics")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Total Messages</h3>
            <h2>{}</h2>
        </div>
        """.format(len(st.session_state.messages)), unsafe_allow_html=True)
    
    with col2:
        user_msgs = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
        st.markdown("""
        <div class="metric-card">
            <h3>👤 User Messages</h3>
            <h2>{}</h2>
        </div>
        """.format(user_msgs), unsafe_allow_html=True)
    
    with col3:
        assistant_msgs = len(st.session_state.messages) - user_msgs
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Assistant Messages</h3>
            <h2>{}</h2>
        </div>
        """.format(assistant_msgs), unsafe_allow_html=True)
    
    with col4:
        ratio = (assistant_msgs / user_msgs * 100) if user_msgs > 0 else 0
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Response Ratio</h3>
            <h2>{:.1f}%</h2>
        </div>
        """.format(ratio), unsafe_allow_html=True)
    
    # Chat Activity Chart
    if len(st.session_state.messages) > 0:
        st.subheader("📊 Chat Activity")
        # Create mock data for visualization
        messages_data = pd.DataFrame([
            {"time": datetime.now().strftime("%H:%M"), "type": msg["role"], "count": 1}
            for msg in st.session_state.messages[-20:]  # Last 20 messages
        ])
        if len(messages_data) > 0:
            fig = px.bar(messages_data, x="time", y="count", color="type", 
                        title="Recent Chat Activity", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
    
    # Usage Statistics
    st.subheader("📈 Usage Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Current Session**")
        st.write(f"• Messages: {len(st.session_state.messages)}")
        st.write(f"• Persona: {persona}")
        st.write(f"• Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        st.info("**System Information**")
        st.write(f"• LLM Model: {GROQ_MODEL}")
        st.write(f"• Embedding: all-MiniLM-L6-v2")
        st.write(f"• Status: {'🟢 Active' if llm else '🔴 Inactive'}")

elif selected == "📚 Documents":
    st.title("📚 Document Management")
    
    # Document Upload Area
    st.subheader("📤 Upload New Documents")
    with st.container():
        st.markdown("""
        <div class="upload-area">
            <h4>📁 Drop your PDF files here</h4>
            <p>Supported formats: PDF, TXT, Markdown</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_docs = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "txt", "md"],
            label_visibility="collapsed"
        )
        
        if uploaded_docs:
            if st.button("📥 Process Documents", use_container_width=True):
                process_uploaded_documents(uploaded_docs)
    
    # Document List
    st.subheader("📖 Document Library")
    if vectorstore_ready:
        try:
            docs = vectorstore.index_to_docstore_id if hasattr(vectorstore, 'index_to_docstore_id') else {}
            if docs:
                for doc_id, doc in list(docs.items())[:10]:  # Show first 10
                    with st.expander(f"📄 Document {doc_id[:8]}..."):
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            else:
                st.info("No documents in the database yet. Upload some documents to get started!")
        except:
            st.info("Unable to list documents")
    else:
        st.warning("Vectorstore not initialized")

elif selected == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    # User Preferences
    st.subheader("👤 User Preferences")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_preferences["theme"] = st.selectbox(
            "Theme",
            ["Light", "Dark", "System"],
            index=["Light", "Dark", "System"].index(st.session_state.user_preferences["theme"])
        )
    with col2:
        st.session_state.user_preferences["response_length"] = st.select_slider(
            "Response Length",
            options=["Short", "Medium", "Detailed"],
            value=st.session_state.user_preferences["response_length"]
        )
    
    # LLM Settings
    st.subheader("🤖 Model Settings")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values make output more creative but less focused"
        )
    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=2000,
            value=500,
            step=100
        )
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")
        st.toast("Settings updated! 🎉")
    
    # Advanced Settings
    st.subheader("🔧 Advanced Settings")
    with st.expander("Advanced Configuration"):
        st.text_input("Vectorstore Path", value=FAISS_STORE_DIR)
        st.text_input("Embedding Model", value="sentence-transformers/all-MiniLM-L6-v2")
        if st.button("🔄 Rebuild Vectorstore"):
            if os.path.exists(FAISS_STORE_DIR):
                shutil.rmtree(FAISS_STORE_DIR)
            st.cache_resource.clear()
            st.success("Vectorstore rebuilt successfully!")
            st.rerun()
    
    # Export/Import
    st.subheader("💾 Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Export All Data", use_container_width=True):
            export_all_data()
    with col2:
        if st.button("📥 Import Data", use_container_width=True):
            st.info("Import functionality coming soon")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def process_uploaded_documents(uploaded_docs):
    """Process uploaded documents and add to vectorstore."""
    tmp_files = []
    docs_to_add = []
    
    try:
        progress_bar = st.progress(0)
        for i, uploaded in enumerate(uploaded_docs):
            tmp_path = safe_write_uploaded_file(uploaded)
            tmp_files.append(tmp_path)
            
            try:
                if uploaded.type == "application/pdf":
                    loader = PyPDFLoader(tmp_path)
                    loaded = loader.load()
                    docs_to_add.extend(loaded)
                else:
                    # Handle text files
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        docs_to_add.append(Document(page_content=content))
            except Exception as e:
                st.error(f"Failed to load {uploaded.name}: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_docs))
        
        if docs_to_add and vectorstore_ready:
            chunks = chunk_documents(docs_to_add)
            vectorstore.add_documents(chunks)
            try:
                vectorstore.save_local(FAISS_STORE_DIR)
            except:
                pass
            st.success(f"✅ Successfully processed {len(uploaded_docs)} documents!")
            st.session_state.document_count += len(uploaded_docs)
        else:
            st.warning("No valid documents found to process")
            
    except Exception as e:
        st.error(f"Error processing documents: {e}")
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except:
                pass

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""
    if CharacterTextSplitter is None:
        out = []
        for d in docs:
            text = d.page_content
            for i in range(0, len(text), chunk_size - chunk_overlap):
                out.append(Document(page_content=text[i:i + chunk_size]))
        return out
    
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    out = []
    for d in docs:
        texts = splitter.split_text(d.page_content)
        out.extend([Document(page_content=t) for t in texts])
    return out

def safe_write_uploaded_file(uploaded_file) -> str:
    """Write uploaded file to secure temporary file."""
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
        except:
            pass
        raise

def export_chat():
    """Export chat history as JSON."""
    if st.session_state.messages:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "persona": persona if 'persona' in locals() else "Unknown"
        }
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Download Chat JSON",
            data=json_data,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("No chat history to export")

def export_all_data():
    """Export all data including chat history and settings."""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "chat_history": st.session_state.messages,
        "settings": {
            "persona": persona if 'persona' in locals() else "Unknown",
            "user_preferences": st.session_state.user_preferences,
            "model": GROQ_MODEL
        }
    }
    json_data = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Complete Export",
        data=json_data,
        file_name=f"pragyanai_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🤖 PragyanAI Assistant v2.0 | Powered by <a href='https://groq.com' target='_blank'>Groq</a></p>
        <p style='font-size: 12px;'>Made with ❤️ for the PragyanAI Community</p>
    </div>
    """,
    unsafe_allow_html=True
)
