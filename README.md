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

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
1. Create a **new GitHub repository** (public or private).
2. Upload these 3 files: `app.py`, `requirements.txt`, `README.md`.
3. Also upload your PDF (`physicology_1st.pdf`) — **not required** since users upload it via the UI.

### Step 2 — Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **"Deploy"** — it will be live in ~2 minutes!

---

## 💻 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

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
