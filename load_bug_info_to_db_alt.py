from pinecone import Pinecone
import os, json, openai, dotenv

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

idx = pc.Index("bug-index")

bugs = json.load(open("bugs.json", encoding="utf-8"))

def embed(txt):
    return openai.embeddings.create(
        input=txt, model="text-embedding-3-small"
    ).data[0].embedding

idx.upsert(vectors = [(b["id"], embed(b["text"]), {"title": b["title"], "text": b["text"]}) for b in bugs])

print("upsert successful")
