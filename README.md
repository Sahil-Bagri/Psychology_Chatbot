# 🧠 Psychology NLP Chatbot — Streamlit App

https://psychologychatbot-hbmr5dpdyhqnmcfajshmrd.streamlit.app/

A retrieval-based chatbot powered by TF-IDF that answers questions from your NCERT Psychology textbook.

---

## 📁 Project Structure

```
psych_streamlit/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---
## 🎯 How to Use

1. Upload your **NCERT Psychology PDF** in the sidebar.
2. Wait a few seconds while the knowledge index builds.
3. Type your question or click a suggested question.
4. Adjust **"Sentences per answer"** and **"Confidence threshold"** sliders to tune responses.

---

## ⚙️ How It Works

| Step | What happens |
|------|-------------|
| PDF Upload | `pdfplumber` extracts all text |
| Tokenization | `nltk.sent_tokenize` splits into sentences |
| Indexing | `TfidfVectorizer` builds a TF-IDF matrix |
| Query | User input is vectorized and compared via cosine similarity |
| Response | Top matching sentences are returned |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `pdfplumber` | PDF text extraction |
| `nltk` | Sentence tokenization |
| `scikit-learn` | TF-IDF vectorizer + cosine similarity |
| `numpy` | Array operations |
