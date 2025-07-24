# 📰 BBC News Summarizer with Semantic Search 

An intelligent news summarization app that:
- Scrapes latest BBC News via RSS
- Extracts full article content using BeautifulSoup
- Generates semantic embeddings using `all-mpnet-base-v2`
- Stores them in a Qdrant vector database
- Performs **semantic search** on user queries
- Summarizes top results using `facebook/bart-large-cnn`
- Built with **FastAPI** and Jinja2 templating

---

## 🧠 How It Works

1. **User enters a query** (e.g. "Ukraine war update")
2. The app:
   - Scrapes BBC news (RSS + article parsing)
   - Vectorizes all articles with SentenceTransformer
   - Stores in Qdrant (semantic vector DB)
   - Encodes user query, searches top-3 similar articles
   - Combines the text, runs BART summarization
3. **Output**: A concise news summary + top articles used

---

## ⚙️ Tech Stack

| Tool / Library            | Purpose                                 |
|---------------------------|------------------------------------------|
| `FastAPI`                 | Web backend framework                    |
| `Jinja2`                  | HTML templating for dynamic rendering    |
| `beautifulsoup4`          | HTML parsing (web scraping)              |
| `requests`                | HTTP requests for scraping               |
| `sentence-transformers`   | Sentence embeddings (`all-mpnet-base-v2`)|
| `qdrant-client`           | Vector DB for semantic similarity        |
| `transformers`            | Text summarization (`facebook/bart-large-cnn`) |
| `torch` / `sentencepiece` | Backend & tokenizer support for models   |

---

## 🚀 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/bbc-news-summarizer.git
cd bbc-news-summarizer
