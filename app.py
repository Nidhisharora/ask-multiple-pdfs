import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# -------------------------------
# MULTI-FORMAT TEXT EXTRACTION
# -------------------------------
def get_all_text(uploaded_files):
    text = ""

    for file in uploaded_files:
        file_extension = file.name.split(".")[-1].lower()

        # ---------- PDF ----------
        if file_extension == "pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        # ---------- DOCX ----------
        elif file_extension == "docx":
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"

        # ---------- TXT / MD ----------
        elif file_extension in ["txt", "md"]:
            text += file.read().decode("utf-8") + "\n"

        else:
            st.warning(f"Unsupported file type: {file.name}")

    return text


# -------------------------------
# TEXT SPLITTING
# -------------------------------
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


# -------------------------------
# VECTOR STORE
# -------------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


# -------------------------------
# ASK LLM
# -------------------------------
def ask_llm(vectorstore, question):

    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No context found."

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not found, say: "Not found in document."

Context:
{context}

Question:
{question}

Answer in 5 lines maximum:
"""

    llm = OllamaLLM(
        model="phi3:mini",
        temperature=0
    )

    return llm.invoke(prompt)


# -------------------------------
# MAIN APP
# -------------------------------
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Documents", page_icon="📚")
    st.title("Chat with Multiple Documents 📚")

    # ---------------- SESSION STATE ----------------
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.subheader("Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, MD files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Reading & indexing documents..."):
                    raw_text = get_all_text(uploaded_files)

                    if not raw_text.strip():
                        st.error("No readable text found.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vectorstore = get_vectorstore(text_chunks)
                        st.success("✅ Documents processed successfully!")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.success("Chat history cleared!")

    # ---------------- DISPLAY CHAT HISTORY ----------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------------- CHAT INPUT ----------------
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:

        if st.session_state.vectorstore is None:
            st.warning("Please upload and process documents first.")
            st.stop()

        # Save user message
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_llm(st.session_state.vectorstore, user_question)
                st.markdown(answer)

        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()
