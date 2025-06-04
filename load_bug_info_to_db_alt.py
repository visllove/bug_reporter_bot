from pinecone import Pinecone
import os, json, openai, dotenv
from pathlib import Path 

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

idx = pc.Index("bug-index")

bug_file= Path(__file__).resolve().parent / "data" / "bugs.json"

with bug_file.open(encoding="utf-8") as f:
    bugs = json.load(f)

def embed(txt):
    return openai.embeddings.create(
        input=txt, model="text-embedding-3-small"
    ).data[0].embedding

idx.upsert(vectors = [(b["id"], embed(b["text"]), {"title": b["title"], "text": b["text"]}) for b in bugs])

print("upsert successful")
