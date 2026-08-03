import os
import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# GROQ API
# -------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
)

# -------------------------------
# PERSONAS
# -------------------------------
SALES_PROMPTS = {

"PragyanAI Student Counselor":
"""
You are Aarav, an Academic & Career Advisor.

Answer ONLY from the context.

Context:
{context}

If answer is not available,
say:
'I couldn't find this information in the uploaded documents.'
""",

"Institution Advisor":
"""
You are an Institutional Relations Lead.

Answer only from the context.

Context:
{context}
""",

"Placement Lead":
"""
You are an Enterprise Placement Lead.

Answer only from context.

Context:
{context}
"""
}

# -------------------------------
# EMBEDDINGS
# -------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------------
# LOAD DOCUMENTS
# -------------------------------

@st.cache_resource
def build_vectorstore():

    docs=[]

    if os.path.exists("pragyan_faq_prices.xlsx"):

        df=pd.read_excel("pragyan_faq_prices.xlsx")

        for _,row in df.iterrows():

            text="\n".join(
                [f"{c}: {row[c]}" for c in df.columns]
            )

            docs.append(
                Document(page_content=text)
            )

    if len(docs)==0:

        docs.append(
            Document(
                page_content="PragyanAI AI Program."
            )
        )

    return FAISS.from_documents(
        docs,
        embeddings
    )

vectorstore=build_vectorstore()

retriever=vectorstore.as_retriever(search_kwargs={"k":4})

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("Settings")

persona=st.sidebar.selectbox(
    "Choose Persona",
    list(SALES_PROMPTS.keys())
)

uploaded_files=st.sidebar.file_uploader(
    "Upload PDFs",
    accept_multiple_files=True,
    type=["pdf"]
)

# -------------------------------
# ADD NEW PDFS
# -------------------------------

if uploaded_files:

    docs=[]

    for file in uploaded_files:

        with open(file.name,"wb") as f:
            f.write(file.read())

        loader=PyPDFLoader(file.name)
        docs.extend(loader.load())

    if docs:

        vectorstore.add_documents(docs)

        st.sidebar.success("Documents Added!")

# -------------------------------
# CHAT HISTORY
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

st.title("🤖 PragyanAI AI Assistant")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt=st.chat_input("Ask anything...")

if prompt:

    st.session_state.messages.append(
        {"role":"user","content":prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    docs=retriever.invoke(prompt)

    context="\n\n".join(
        [d.page_content for d in docs]
    )

    system_prompt=SALES_PROMPTS[persona].format(
        context=context
    )

    response=llm.invoke(
        system_prompt + "\n\nUser: " + prompt
    )

    answer=response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )
