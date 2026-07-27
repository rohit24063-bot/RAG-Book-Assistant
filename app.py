import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚"
)

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document.")


# ---------------------- Cached Resources ----------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return init_chat_model(
        "google_genai:gemini-3.1-flash-lite",
        temperature=0.1,
        max_tokens=500
    )


# ---------------------- Upload PDF ----------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Creating embeddings... Please wait."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = load_embeddings()

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore

        try:
            os.remove(file_path)
        except OSError:
            pass

        st.success("Vector database created successfully!")


# ---------------------- Question Answering ----------------------

if "vectorstore" in st.session_state:

    vectorstore = st.session_state.vectorstore

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = load_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not contained in the context,
reply exactly:

"I could not find the answer in the uploaded document."

Do not hallucinate.
Keep your answers concise.
"""
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.chat_input("Enter your question")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        response = llm.invoke(final_prompt)

        st.write("### AI Answer")
        st.write(response.text)

        st.write("---")
        st.write("### Sources")

        pages = []

        for doc in docs:
            page = doc.metadata.get("page")
            if page is not None:
                pages.append(page + 1)

        pages = sorted(set(pages))

        if pages:
            st.write(
                "Relevant pages: "
                + ", ".join(str(p) for p in pages)
            )