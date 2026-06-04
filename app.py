import streamlit as st
import pdfplumber
import nltk
import numpy as np
import re
import os

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── NLTK downloads (silent) ──────────────────────────────────────────────────
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Psychology Chatbot",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    .chat-bubble-user {
        background: #4f8ef7;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 78%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-bubble-bot {
        background: #ffffff;
        color: #1a1a2e;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        max-width: 78%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        font-size: 0.95rem;
    }
    .chat-container { padding: 10px 0; }
    .source-badge {
        font-size: 0.72rem;
        color: #888;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,?!\'\-]", "", text)
    return text.strip()


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from an uploaded PDF file object."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text


def build_corpus(text: str):
    """Tokenize text into clean sentences."""
    raw = sent_tokenize(text)
    corpus = [s.strip() for s in raw if len(s.split()) > 8]
    return corpus


def build_index(corpus):
    """Fit TF-IDF vectorizer on corpus."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def get_response(user_input: str, vectorizer, matrix, corpus,
                 top_k: int = 2, threshold: float = 0.10) -> tuple[str, float]:
    """Return (answer_text, confidence_score)."""
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    sims = cosine_similarity(vec, matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    best = sims[top_idx[0]]

    if best < threshold:
        return ("I'm not sure about that based on the textbook. "
                "Try rephrasing or ask about psychology topics like "
                "behaviourism, Gestalt, psychoanalysis, or branches of psychology."), best

    seen, sentences = set(), []
    for idx in top_idx:
        if sims[idx] > threshold:
            s = corpus[idx].strip()
            if s not in seen:
                sentences.append(s)
                seen.add(s)

    return " ".join(sentences), round(float(best), 3)


# ── Session state initialisation ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, conf}
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "matrix" not in st.session_state:
    st.session_state.matrix = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# ── Sidebar — PDF upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
             width=60)  # placeholder — replace with your own logo if desired
    st.title("🧠 Setup")
    st.markdown("Upload your **NCERT Psychology PDF** to power the chatbot.")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and not st.session_state.pdf_loaded:
        with st.spinner("Reading PDF and building knowledge index…"):
            try:
                text   = extract_text_from_pdf(uploaded)
                corpus = build_corpus(text)
                if len(corpus) < 10:
                    st.error("Too few sentences extracted. Check your PDF.")
                else:
                    vec, mat = build_index(corpus)
                    st.session_state.vectorizer = vec
                    st.session_state.matrix     = mat
                    st.session_state.corpus     = corpus
                    st.session_state.pdf_loaded = True
                    st.success(f"✅ Ready! {len(corpus)} sentences indexed.")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")

    if st.session_state.pdf_loaded:
        st.markdown(f"**Sentences indexed:** {len(st.session_state.corpus)}")
        st.markdown("---")
        top_k = st.slider("Sentences per answer", 1, 4, 2)
        threshold = st.slider("Confidence threshold", 0.05, 0.50, 0.10, 0.01)

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        top_k = 2
        threshold = 0.10

    st.markdown("---")
    st.caption("Built with Streamlit · TF-IDF · pdfplumber")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🧠 Psychology Chatbot")
st.caption("Ask anything from your NCERT Psychology textbook.")

if not st.session_state.pdf_loaded:
    st.info("👈 Upload your psychology PDF in the sidebar to get started.")
    st.stop()

# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble-user">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-bubble-bot">{msg["content"]}'
            f'<div class="source-badge">Confidence: {msg.get("conf", "—")}</div></div>',
            unsafe_allow_html=True,
        )

# ── Input box ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Suggested questions
with st.expander("💡 Suggested questions", expanded=False):
    suggestions = [
        "What is psychology?",
        "Who founded behaviourism?",
        "What is Gestalt psychology?",
        "What are the branches of psychology?",
        "How did psychology develop in India?",
        "What is introspection?",
        "What is psychoanalysis?",
        "What is the difference between mind and brain?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f"sugg_{i}"):
            st.session_state["prefill"] = q
            st.rerun()

# Pre-fill from suggestion click
default_val = st.session_state.pop("prefill", "")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question",
        value=default_val,
        placeholder="e.g. What is behaviourism?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Send ➤")

if submitted and user_input.strip():
    q = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": q})

    answer, conf = get_response(
        q,
        st.session_state.vectorizer,
        st.session_state.matrix,
        st.session_state.corpus,
        top_k=top_k,
        threshold=threshold,
    )
    st.session_state.messages.append({"role": "bot", "content": answer, "conf": conf})
    st.rerun()
