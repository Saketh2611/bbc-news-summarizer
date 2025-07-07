# 📰 BBC News Summarizer with Vector Search

A Streamlit app that:
- Scrapes BBC News via RSS
- Extracts full article content
- Generates embeddings using SentenceTransformer (MPNet)
- Stores them in Qdrant vector database
- Performs semantic search based on a user query
- Summarizes top results using `facebook/bart-large-cnn`

---

## 🔧 Tech Stack

| Tool / Library         | Purpose                                 |
|------------------------|------------------------------------------|
| `streamlit`            | UI framework                             |
| `beautifulsoup4`       | HTML parsing (web scraping)              |
| `requests`             | Sending HTTP requests                    |
| `sentence-transformers`| Generating vector embeddings             |
| `qdrant-client`        | Vector database for similarity search    |
| `transformers`         | Text summarization using BART model      |
| `torch`                | PyTorch backend for models               |
| `sentencepiece`        | Tokenizer support                        |

---

## 🚀 How to Run the App Locally

### 1. Install Dependencies

If you have a `requirements.txt` file: (download from Above)

```bash
pip install -r requirements.txt
streamlit run app_v.py

