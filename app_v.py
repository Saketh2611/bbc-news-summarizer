import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# === CONFIG ===
collection_name = "news_articles"
qdrant_url = "https://23a37241-1707-4f1a-8f5e-47c00502551d.us-west-1-0.aws.cloud.qdrant.io:6333"
qdrant_api_key = "Your Api key"

# === FUNCTIONS ===
@st.cache_data(show_spinner=False)
def scrape_bbc_rss(limit=10):
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
    else :
        client.recreate_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(size=768 , distance=Distance.COSINE)
        )

    points = [
        PointStruct(id=i, vector=embeddings[i], payload=articles[i])
        for i in range(len(articles))
    ]

    client.upsert(collection_name=collection_name, points=points)

# === LOAD MODELS ===
@st.cache_resource(show_spinner=True)
def load_models():
    embed_model = SentenceTransformer('all-mpnet-base-v2')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return embed_model, summarizer

# === STREAMLIT UI ===
st.title("🇧 🇧 🇨 Smart News Summarizer")

embed_model, summarizer = load_models()

st.sidebar.header("Search News")
query = st.sidebar.text_input("Enter your query", "Gaza ceasefire updates")
if st.sidebar.button("Search"):
    with st.spinner("Scraping latest news..."):
        articles = scrape_bbc_rss()

    embeddings = embed_model.encode(
        [f"{a['title']}. {a['text']}" for a in articles],
        normalize_embeddings=True
    )

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    setup_qdrant(client, embeddings, articles)

    query_vector = embed_model.encode(query, normalize_embeddings=True)

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )

    st.subheader("Top Matching Articles")
    context = ""
    for i, hit in enumerate(search_result):
        article = hit.payload
        st.markdown(f"**{i+1}. {article['title']}**")
        st.markdown(f"[Read More]({article['url']})")
        st.markdown(article['text'][:400] + "...")
        context += article['text'] + "\n\n"

    st.subheader("🧾 Summary")
    if len(context.split()) > 1000:
        context = " ".join(context.split()[:1000])
    summary = summarizer(context, max_length=250, min_length=80, do_sample=False)[0]['summary_text']
    st.success(summary)
