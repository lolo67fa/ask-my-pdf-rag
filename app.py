"""Streamlit Web Application for 'Ask My PDF'.

A clean, beginner-friendly interface with purple accents that:
1. Allows uploading a single PDF.
2. Extracts pages with 1-based page numbers.
3. Splits text into overlapping chunks.
4. Retrieves top 3 chunks with TF-IDF cosine similarity.
5. Uses Gemini function calling for document statistics and grounded Q&A with page citations.
"""

import os
import base64
import html
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import extract_pages_from_pdf, chunk_pages, compute_document_metrics
from retriever import TFIDFRetriever
from gemini_service import (
    register_document,
    clear_document_registry,
    get_document_stats,
    answer_question_with_source,
    SOURCE_GEMINI,
    SOURCE_OFFLINE,
    SOURCE_FALLBACK,
    SOURCE_API_ERROR,
)

# Load environment variables
load_dotenv()

SOURCE_LABELS = {
    SOURCE_GEMINI: "🤖 Answered by Gemini 2.5 Flash",
    SOURCE_OFFLINE: "💻 Answered by the local offline engine (Gemini not configured)",
    SOURCE_FALLBACK: "💻 Answered by the local offline engine (Gemini was unavailable)",
    SOURCE_API_ERROR: "⚠️ Gemini API error (no answer generated)",
}

MODE_LABELS = {
    "summary": "Summary Path",
    "full_document": "Full Document (small PDF)",
    "tfidf": "TF-IDF",
}


def resolve_gemini_key() -> str:
    """Read GEMINI_API_KEY from the environment/.env, then Streamlit secrets. Never displayed."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            # No secrets.toml configured (normal for local runs)
            key = ""
    return key


def render_chunks(chunks):
    """Show retrieved passages with page, chunk id, retrieval mode, and score."""
    with st.expander("🔍 View Retrieved Document Chunks"):
        for c in chunks:
            mode_label = MODE_LABELS.get(c.get("retrieval_mode", "tfidf"), "TF-IDF")
            st.markdown(
                f"**Page {c.get('page_number', '?')}** | Chunk `{c.get('chunk_id', '')}` | "
                f"Mode: `{mode_label}` | Score: `{c.get('score', 0.0):.4f}`\n\n"
                f"> {c.get('text', '')}"
            )
            st.divider()

# Page configuration
st.set_page_config(
    page_title="Ask My PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Purple accent custom styling
st.markdown("""
<style>
    :root {
        --primary-purple: #7C3AED;
        --primary-light: #EDE9FE;
        --primary-dark: #5B21B6;
    }
    
    /* Main titles and header accents */
    .purple-title {
        color: #7C3AED;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .purple-subtitle {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Citation badges */
    .citation-badge {
        display: inline-block;
        background-color: #EDE9FE;
        color: #6D28D9;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        margin-left: 0.25rem;
        border: 1px solid #DDD6FE;
    }

    /* Stat card */
    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .stat-card-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stat-card-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #7C3AED;
    }
    
    /* Custom button styles */
    .stButton>button {
        border-radius: 8px;
        border-color: #7C3AED;
    }
    .stButton>button:hover {
        border-color: #5B21B6;
        color: #5B21B6;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_metrics" not in st.session_state:
    st.session_state.doc_metrics = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Sidebar
with st.sidebar:
    st.markdown('<div class="purple-title" style="font-size: 1.7rem;">📄 Ask My PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="purple-subtitle">Grounded PDF Q&A with TF-IDF & Gemini</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Connection Status
    st.subheader("🤖 AI Connection")
    gemini_key = resolve_gemini_key()
    is_gemini_configured = bool(gemini_key and gemini_key != "your_gemini_api_key_here")

    if is_gemini_configured:
        st.success("Gemini API key detected (connection is checked on first question)")
    else:
        st.warning("Gemini not configured: running in offline simulation mode")

    st.markdown("---")
    
    # PDF Upload
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Select a PDF file",
        type=["pdf"],
        help="Upload a single PDF to extract pages and query."
    )

    # Process uploaded file if new or changed
    if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
        with st.spinner("Parsing document and computing TF-IDF index..."):
            try:
                pdf_bytes = uploaded_file.read()
                pages = extract_pages_from_pdf(pdf_bytes)
                chunks = chunk_pages(pages, chunk_size=150, chunk_overlap=30)
                metrics = compute_document_metrics(pages)

                doc_id = uploaded_file.name
                clear_document_registry()
                register_document(doc_id, metrics["page_count"], metrics["word_count"])

                st.session_state.retriever = TFIDFRetriever(chunks)
                st.session_state.chunks = chunks
                st.session_state.doc_id = doc_id
                st.session_state.doc_metrics = metrics
                st.session_state.last_uploaded_file = doc_id
                st.session_state.messages = []  # Reset chat for new document

                st.toast(f"Successfully processed {doc_id}!", icon="✅")
            except ValueError as val_err:
                st.error(f"⚠️ Document processing error: {val_err}")
                st.session_state.retriever = None
                st.session_state.doc_id = None
            except Exception as err:
                st.error(f"❌ Unexpected error reading PDF: {err}")
                st.session_state.retriever = None
                st.session_state.doc_id = None

    # Document metrics card
    if st.session_state.doc_metrics:
        st.markdown("---")
        st.subheader("📊 Document Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Pages</div>
                <div class="stat-card-value">{st.session_state.doc_metrics['page_count']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-title">Words</div>
                <div class="stat-card-value">{st.session_state.doc_metrics['word_count']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"Indexed **{len(st.session_state.chunks)}** overlapping chunks with 1-based page metadata.")

        # Test tool button
        if st.button("🔧 Test 'get_document_stats' Tool"):
            clear_document_registry()
            register_document(
                st.session_state.doc_id,
                st.session_state.doc_metrics["page_count"],
                st.session_state.doc_metrics["word_count"],
            )
            stats = get_document_stats(st.session_state.doc_id)
            st.json(stats)

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Layout - Responsive Header Row
header_col, logo_col = st.columns([3, 1])

with header_col:
    st.markdown('<div class="purple-title">Ask My PDF</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="purple-subtitle">Ask questions strictly grounded in your document with verifiable page citations.</div>',
        unsafe_allow_html=True
    )

with logo_col:
    logo_path = Path(__file__).resolve().parent / "Sdaia-Academy.png"
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as img_file:
                logo_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 1rem;">
                    <div style="background-color: #FFFFFF; width: 230px; height: 100px; padding: 6px 14px; border-radius: 10px; border: 1px solid #DDD6FE; box-shadow: 0 2px 10px rgba(124, 58, 237, 0.12); display: flex; align-items: center; justify-content: center;">
                        <img src="data:image/png;base64,{logo_b64}" alt="SDAIA Academy Logo" style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; display: block;" />
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            pass

# If no document is uploaded yet, show welcome card
if not st.session_state.doc_id:
    st.markdown("""
    <div style="background-color: #FAF5FF; border: 1px solid #E9D5FF; border-radius: 12px; padding: 24px; margin-top: 10px;">
        <h3 style="color: #6B21A8; margin-top: 0;">👋 Welcome to Ask My PDF</h3>
        <p style="color: #4B5563; font-size: 1.05rem;">
            Ask questions about a PDF and get answers grounded in retrieved excerpts with page citations. Always verify important answers against the cited pages.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 16px;">
            <div style="background: white; padding: 16px; border-radius: 8px; border: 1px solid #F3E8FF;">
                <strong style="color: #7C3AED;">1. Page-Preserved Extraction</strong>
                <p style="color: #6B7280; font-size: 0.9rem; margin-top: 4px;">PyMuPDF extracts each page preserving strict 1-based page numbers.</p>
            </div>
            <div style="background: white; padding: 16px; border-radius: 8px; border: 1px solid #F3E8FF;">
                <strong style="color: #7C3AED;">2. TF-IDF Cosine Retrieval</strong>
                <p style="color: #6B7280; font-size: 0.9rem; margin-top: 4px;">Scikit-learn retrieves the top 3 most relevant overlapping excerpts without external vector databases.</p>
            </div>
            <div style="background: white; padding: 16px; border-radius: 8px; border: 1px solid #F3E8FF;">
                <strong style="color: #7C3AED;">3. Grounded Gemini Answers</strong>
                <p style="color: #6B7280; font-size: 0.9rem; margin-top: 4px;">Gemini is instructed to answer only from excerpts with <code>[Page X]</code> citations and to refuse when evidence is missing.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Upload a PDF in the sidebar to get started!")
else:
    # Active document banner
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background-color: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 8px; padding: 10px 16px; margin-bottom: 20px;">
        <div>
            <span style="font-weight: 700; color: #5B21B6;">📄 Active Document:</span>
            <span style="color: #4C1D95; margin-left: 6px;">{html.escape(st.session_state.doc_id)}</span>
        </div>
        <div>
            <span class="citation-badge">{st.session_state.doc_metrics['page_count']} Pages</span>
            <span class="citation-badge">{st.session_state.doc_metrics['word_count']} Words</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        chunks = msg.get("chunks", [])

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant" and msg.get("source"):
                st.caption(SOURCE_LABELS.get(msg["source"], ""))

            # If there are retrieved chunks associated with this assistant message, display them in an expander
            if role == "assistant" and chunks:
                render_chunks(chunks)

    # Chat input box
    prompt = st.chat_input("Ask a question about the document...")
    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve top 3 chunks using TF-IDF
        top_chunks = st.session_state.retriever.retrieve(prompt, top_k=3)

        # The tool registry is shared by the server process; make sure it holds this session's document
        clear_document_registry()
        register_document(
            st.session_state.doc_id,
            st.session_state.doc_metrics["page_count"],
            st.session_state.doc_metrics["word_count"],
        )

        # Generate answer with Gemini or offline mode
        with st.chat_message("assistant"):
            with st.spinner("Finding relevant passages & generating grounded answer..."):
                answer, source = answer_question_with_source(
                    question=prompt,
                    chunks=top_chunks,
                    doc_id=st.session_state.doc_id,
                    api_key=gemini_key or None,
                )
                st.markdown(answer)
                st.caption(SOURCE_LABELS.get(source, ""))

                if top_chunks:
                    render_chunks(top_chunks)

        # Save assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks": top_chunks,
            "source": source,
        })
