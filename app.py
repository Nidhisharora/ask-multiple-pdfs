import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import time
import re
# ---------------- SETTINGS ----------------
st.set_page_config(
    page_title="Smart Answer Extractor", 
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="collapsed"
)

# Custom CSS for centered layout and premium gradient
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Premium deep gradient background */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1a1f35 0%, #0d0f1c 50%, #0a0b14 100%) !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Add subtle floating particles effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(192, 132, 252, 0.03) 0%, transparent 50%);
        pointer-events: none;
    }
    
    /* Center content container */
    .centered-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem 2rem;
        position: relative;
        z-index: 1;
    }
    
    /* Main header with premium gradient */
    .main-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    .main-header h1 {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #f0a6ca 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .main-header p {
        color: #9ca3af;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Premium card styling */
    .premium-card {
        background: rgba(20, 25, 45, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        margin: 1.5rem 0;
    }
    
    /* Center file uploader */
    .uploader-wrapper {
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 20, 35, 0.6) !important;
        border: 2px dashed rgba(167, 139, 250, 0.3) !important;
        border-radius: 28px !important;
        padding: 3rem 2rem !important;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #a78bfa !important;
        background: rgba(25, 30, 50, 0.8) !important;
        box-shadow: 0 0 30px rgba(167, 139, 250, 0.2) !important;
    }
    
    /* File uploader text */
    div[data-testid="stFileUploader"] > section {
        color: #e2e8f0 !important;
    }
    
    /* Browse files button */
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 60px !important;
        font-weight: 500 !important;
        margin: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(167, 139, 250, 0.4) !important;
    }
    
    /* Center text input */
    .input-wrapper {
        max-width: 600px;
        margin: 2rem auto;
    }
    
    /* Question input styling */
    div[data-testid="stTextInput"] {
        max-width: 600px;
        margin: 0 auto;
    }
    
    div[data-testid="stTextInput"] > div > div > input {
        background: rgba(15, 20, 35, 0.8) !important;
        border: 2px solid rgba(167, 139, 250, 0.2) !important;
        border-radius: 60px !important;
        padding: 1.2rem 2rem !important;
        color: white !important;
        font-size: 1.1rem !important;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 4px rgba(167, 139, 250, 0.15) !important;
        background: rgba(20, 25, 45, 0.9) !important;
    }
    
    div[data-testid="stTextInput"] > div > div > input::placeholder {
        color: #6b7280 !important;
        text-align: center;
    }
    
    /* Center button */
    .button-wrapper {
        max-width: 300px;
        margin: 2rem auto;
    }
    
    /* Button styling */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 50%, #fb7185 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
        border-radius: 60px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(167, 139, 250, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 40px rgba(167, 139, 250, 0.5) !important;
    }
    
    /* Stats cards centered */
    .stats-container {
        max-width: 600px;
        margin: 2rem auto;
    }
    
    div[data-testid="column"] {
        background: rgba(15, 20, 35, 0.6) !important;
        backdrop-filter: blur(10px);
        border-radius: 24px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        text-align: center;
    }
    
    /* Numbers in stats */
    div[data-testid="column"] p:first-child {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem !important;
    }
    
    /* Answer container centered */
    .answer-wrapper {
        max-width: 700px;
        margin: 2rem auto;
    }
    
    .answer-box {
        background: rgba(15, 20, 35, 0.8) !important;
        backdrop-filter: blur(10px);
        border-radius: 28px !important;
        padding: 2rem !important;
        border-left: 4px solid #a78bfa !important;
        border: 1px solid rgba(167, 139, 250, 0.2);
    }
    
    .answer-box h4 {
        color: #a78bfa !important;
        margin-bottom: 1rem !important;
        font-size: 1.2rem !important;
    }
    
    .answer-box p {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
    }
    
    /* Source expander */
    div[data-testid="stExpander"] {
        background: rgba(15, 20, 35, 0.6) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Success message centered */
    .stAlert {
        max-width: 600px !important;
        margin: 1rem auto !important;
        background: rgba(16, 185, 129, 0.1) !important;
        color: #10b981 !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 60px !important;
        padding: 1rem 2rem !important;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    /* Footer - always at bottom */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 1.5rem;
        background: rgba(10, 12, 20, 0.8);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        font-size: 0.95rem;
        z-index: 100;
    }
    
    .footer p {
        margin: 0;
        color: #94a3b8 !important;
    }
    
    .footer .heart {
        color: #f472b6;
        display: inline-block;
        animation: heartbeat 1.5s ease infinite;
    }
    
    .footer .tech-stack {
        color: #a78bfa;
        font-weight: 500;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    /* Add padding to prevent content from hiding behind footer */
    .main-content {
        padding-bottom: 100px;
    }
    
    /* Progress bar */
    div[data-testid="stProgress"] > div {
        background-color: rgba(30, 41, 59, 0.5) !important;
    }
    
    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%) !important;
    }
    
    /* Make all text visible */
    p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #a78bfa !important;
        border-top-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LANDING PAGE MODE ----------------
if 'show_app' not in st.session_state:
    st.session_state.show_app = False

if not st.session_state.show_app:
    # Hide Streamlit elements for cleaner look
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp { margin-top: -80px; }
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    try:
        with open("landing_page.html", "r", encoding="utf-8") as f:
            landing_html = f.read()
        
        # Inject the HTML
        st.components.v1.html(landing_html, height=800, scrolling=True)
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Launch Smart Answer Extractor", use_container_width=True):
                st.session_state.show_app = True
                st.rerun()
                
    except FileNotFoundError:
        st.error("⚠️ Landing page file not found.")
        if st.button("Continue to App anyway"):
            st.session_state.show_app = True
            st.rerun()
            
    except UnicodeDecodeError:
        try:
            with open("landing_page.html", "r", encoding="latin-1") as f:
                landing_html = f.read()
            st.components.v1.html(landing_html, height=800, scrolling=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚀 Launch Smart Answer Extractor", use_container_width=True):
                    st.session_state.show_app = True
                    st.rerun()
        except:
            st.error("Could not read the landing page file.")
        
else:
    # ---------------- MAIN CONTENT WRAPPER ----------------
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # ---------------- MAIN HEADER ----------------
    st.markdown("""
    <div class="centered-container">
        <div class="main-header">
            <h1>📚 Smart Answer Extractor</h1>
            <p>Transform your exam notes into intelligent answers instantly</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- LOAD MODEL ----------------
    @st.cache_resource
    def load_model():
        with st.spinner("🚀 Loading AI model..."):
            time.sleep(1)
            return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    model = load_model()
    
    # ---------------- FUNCTIONS ----------------
    def read_pdf(file):
        reader = PdfReader(file)
        text_pages = []
        progress_bar = st.progress(0)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                text_pages.append((i+1, text))
            progress_bar.progress((i + 1) / len(reader.pages))
        progress_bar.empty()
        return text_pages
    
    def split_text(text, chunk_size=400, overlap=80):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    

    
    def create_embeddings(chunks):

        # Ensure chunks is a flat list of strings
        cleaned_chunks = [
            str(chunk) for chunk in chunks
            if chunk is not None and isinstance(chunk, (str, int, float))
        ]

        print(f"Total chunks after cleaning: {len(cleaned_chunks)}")
        print(f"Sample chunk type: {type(cleaned_chunks[0])}")

        embeddings = model.encode(cleaned_chunks, show_progress_bar=True)
        return embeddings
        

    
    def search_answer(question, chunks, index, k=5, threshold=0.65):
        q_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        D, I = index.search(q_embedding, k)

        similarities = 1 - D[0]  # Convert L2 to similarity

        valid_chunks = []
        for idx, score in zip(I[0], similarities):
            if score > threshold:
                valid_chunks.append(chunks[idx])

        return valid_chunks
    
       


    
    def ask_llm(context, question):
        prompt = f"""
    You are a strict document-based answer extractor.

    Rules:
    - Use ONLY the information present in the context.
    - Do NOT use external knowledge.
    - If answer is not clearly mentioned,
    respond ONLY with:
    "Answer not found in notes."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

        response = ollama.chat(
            model="tinyllama",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        return response["message"]["content"]

    # ---------------- SESSION ----------------
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "pages" not in st.session_state:
        st.session_state.pages = []
    
    # ---------------- CENTERED UPLOAD SECTION ----------------
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Notes")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload your exam notes in PDF format (Max 200MB)"
    )
    
    if uploaded_file:
        with st.spinner("🔄 Processing your notes..."):
            pages = read_pdf(uploaded_file)
            all_chunks = []
            page_map = []
    
            for page_no, text in pages:
                chunks = split_text(text)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    page_map.append(page_no)
    
            embeddings = create_embeddings(all_chunks)
    
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
    
            st.session_state.chunks = all_chunks
            st.session_state.index = index
            st.session_state.pages = page_map
    
            st.markdown("""
            <div class="stAlert">
                ✅ Notes processed successfully! You can now ask questions.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------------- STATS SECTION ----------------
    if st.session_state.chunks:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**{len(st.session_state.chunks)}**  \nText Chunks")
        
        with col2:
            unique_pages = len(set(st.session_state.pages))
            st.markdown(f"**{unique_pages}**  \nPages")
        
        with col3:
            avg_chunk_size = np.mean([len(c.split()) for c in st.session_state.chunks])
            st.markdown(f"**{int(avg_chunk_size)}**  \nAvg Words/Chunk")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------------- CENTERED QUESTION SECTION ----------------
    if st.session_state.chunks:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### ❓ Ask Your Question")
        
        question = st.text_input(
            "What would you like to know from your notes?",
            placeholder="e.g., What is the capital of France?",
            label_visibility="collapsed" 
        )
        
        search_clicked = st.button("🔍 Find Answer", use_container_width=True)

        if search_clicked:
            if not question.strip():
                st.warning("⚠️ Please enter a question first.")
            else:
                answer = None
                best_chunks = []

                with st.spinner("🤔 Searching for the best answer..."):

                    if "overview" in question.lower() or "summary" in question.lower():
                        full_text = "\n\n".join(st.session_state.chunks[:30])
                        answer = ask_llm(full_text, "Summarize this document.")
                    else:
                        best_chunks = search_answer(
                            question,
                            st.session_state.chunks,
                            st.session_state.index
                        )

                        if not best_chunks:
                            answer = "Answer not found in notes."
                        else:
                            context = "\n\n".join(best_chunks)[:1200]
                            answer = ask_llm(context, question)

                    st.session_state.last_question = question
                    st.session_state.last_answer = answer


                # Display Answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("#### ✅ Answer:")
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                # Source Text
                with st.expander("📖 View Source Text"):
                    st.markdown(
                        '<div style="background: rgba(15,20,35,0.6); padding: 1rem; border-radius: 20px;">',
                        unsafe_allow_html=True
                    )
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)

        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-content

    # ---------------- FIXED FOOTER ----------------
    st.markdown("""
    <div class="footer">
        <p>Built with <span class="heart">❤️</span> for students | <span class="tech-stack">Offline AI Project</span></p>
        <p style="font-size: 0.8rem; margin-top: 0.3rem;">Powered by Streamlit • FAISS • Ollama</p>
    </div>
    """, unsafe_allow_html=True)