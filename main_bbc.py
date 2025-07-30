from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Qdrant Config ===
collection_name = "news_articles_1"
qdrant_url = "https://23a37241-1707-4f1a-8f5e-47c00502551d.us-west-1-0.aws.cloud.qdrant.io:6333"
qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6MHdGWXVS2dEszyAaokzSlQbqe0Fdh_vFEvBJxXH50c"

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# === Load Models at Startup ===
embed_model = SentenceTransformer("all-mpnet-base-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# === Helper Functions ===
def scrape_bbc_rss(limit=100):
    url = 'https://feeds.bbci.co.uk/news/rss.xml'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    items = soup.find_all('item')[:limit]

    articles = []
    for item in items:
        title = item.title.text
        link = item.link.text
        try:
            article_res = requests.get(link, headers={'User-Agent': 'Mozilla/5.0'})
            article_soup = BeautifulSoup(article_res.content, 'html.parser')
            paragraphs = article_soup.select('article p')
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        except:
            text = ""

        articles.append({
            'title': title,
            'url': link,
            'text': text[:1000],
            'published_date': datetime.now().isoformat(),
            'category': 'bbc'
        })
    return articles


def setup_qdrant(client, embeddings, articles):
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    else:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    points = [
        PointStruct(id=i, vector=embeddings[i], payload=articles[i])
        for i in range(len(articles))
    ]
    client.upsert(collection_name=collection_name, points=points)


# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("BBC.html", {"request": request})


@app.post("/summarize", response_class=HTMLResponse)
async def summarize_news(request: Request, query: str = Form(...)):
    articles = scrape_bbc_rss()
    embeddings = [embed_model.encode(f"{a['title']}. {a['text']}", normalize_embeddings=True) for a in articles]

    setup_qdrant(client, embeddings, articles)
    query_vector = embed_model.encode(query, normalize_embeddings=True)

    hits = client.search(collection_name=collection_name, query_vector=query_vector, limit=3)

    top_articles = [hit.payload for hit in hits]
    context = "\n\n".join([a["text"] for a in top_articles])

    # Truncate context if too long
    if len(context.split()) > 1000:
        context = " ".join(context.split()[:1000])

    prompt = f"Answer the following question based on the news articles:\n\nQuestion: {query}\n\nArticles:\n{context}"
    summary = summarizer(prompt, max_length=250, min_length=80, do_sample=False)[0]["summary_text"]


    return templates.TemplateResponse("BBC.html", {
        "request": request,
        "query": query,
        "articles": top_articles,
        "summary": summary
    })



