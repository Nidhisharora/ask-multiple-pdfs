import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# -------------------------------
# TEXT SPLITTING (Fast + Stable)
# -------------------------------
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=220,        
        chunk_overlap=20,
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
# CONVERSATION CHAIN (Optimized)
# -------------------------------
def ask_llm(vectorstore, question):

    docs = vectorstore.similarity_search(question, k=1)
    context = docs[0].page_content if docs else "No context found."

    prompt = f"""
        You are a helpful assistant.
        Answer ONLY from the context below.
        If answer not present, say: Not found in document.

        Context:
        {context}

        Question:
        {question}

        Answer in 5 lines maximum:
        """

    llm = Ollama(
        model="phi3:mini",
        temperature=0,
        num_ctx=256,
        num_predict=80
    )

    return llm.invoke(prompt)
# -------------------------------
# HANDLE USER INPUT
# -------------------------------
def handle_userinput(user_question):
    user_question = user_question[:150]

    if st.session_state.vectorstore is None:
        st.warning("Please upload and process documents first.")
        return

    answer = ask_llm(st.session_state.vectorstore, user_question)

    st.markdown(f"**You:** {user_question}")
    st.markdown(f"**Bot:** {answer}")


# -------------------------------
# MAIN APP
# -------------------------------
def main():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")

    st.header("Chat with multiple PDFs 📚")

    # Session State Setup
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processed" not in st.session_state:
        st.session_state.processed = False

    # User Question Input
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.subheader("Your documents")
    
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True
        )
    
        # PROCESS BUTTON
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return
    
            with st.spinner("Reading & indexing PDF... (20-40 sec first time)"):
            
                raw_text = get_pdf_text(pdf_docs)
    
                if not raw_text.strip():
                    st.error("No readable text found in PDFs.")
                    return
    
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
    
                st.session_state.vectorstore = vectorstore
                st.success("✅ Ready! Ask questions instantly now.")
    
        # CLEAR CHAT
        if st.button("Clear Chat"):
            st.session_state.vectorstore = None
            st.success("Cleared! Upload & Process again.")

if __name__ == "__main__":
    main()