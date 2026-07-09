import re
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


OPENSTAX_URLS = [
    "https://openstax.org/books/principles-data-science/pages/7-3-introduction-to-deep-learning",
    "https://openstax.org/books/principles-data-science/pages/7-4-neural-networks",
    "https://openstax.org/books/principles-data-science/pages/7-5-natural-language-processing",
]


def fetch_openstax_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    # remove nav, header, footer noise
    for tag in soup(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    # grab main content
    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n")

    # clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class LectureNoteRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 3):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.passages = []
        self.index = None

    def build_index(self) -> None:
        print("Fetching OpenStax content...")
        all_chunks = []
        for url in OPENSTAX_URLS:
            print(f"  Fetching {url.split('/')[-1]}...")
            text = fetch_openstax_text(url)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks")

        self.passages = all_chunks
        print(f"Encoding {len(self.passages)} chunks...")
        embeddings = self.model.encode(self.passages, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print("Index built.")

    def retrieve(self, query: str) -> list[str]:
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        distances, indices = self.index.search(query_embedding, self.top_k)
        return [self.passages[i] for i in indices[0] if i < len(self.passages)]