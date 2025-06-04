import os, json, logging, time, dotenv, hashlib
from pinecone import Pinecone, exceptions

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

PC_KEY  = os.getenv("PINECONE_API_KEY")
ENV     = os.getenv("PINECONE_ENV")
INDEX   = "bugs-index"
MODEL   = "llama-text-embed-v2"         

pc = Pinecone(api_key=PC_KEY)

# ensure index exists (serverless + integrated)

try:
    pc.describe_index(INDEX)
    logging.info(f"Index {INDEX} already exists", )
except exceptions.NotFoundException:
    logging.info(f"Creating serverless index {INDEX} for model {MODEL} …")
    pc.create_index_for_model(
        name=INDEX,
        model=MODEL,
        dimension=None,               
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": ENV.split("-")[0]}},
        field_map={"text": "text"}  
    )

index = pc.Index(INDEX)


# check json with bugs before load

def ascii_id(s: str) -> str:
    """Generate ASCII‑safe id if original contains non‑ASCII"""
    return hashlib.sha1(s.encode()).hexdigest()

def load_bugs(path: str = "bugs.json"):
    bugs = json.load(open(path, encoding="utf-8"))
    out = []
    for b in bugs:
        _id = b.get("id") or ascii_id(b["title"])
        if not _id.isascii():
            _id = ascii_id(_id)
        if not b["text"]:
            logging.warning(f"Skip bug {_id}: empty text", )
            continue
        out.append({"_id": _id, "title": b["title"], "text": b["text"]})
    return out

records = load_bugs()
logging.info(f"Validated {len(records)} bugs", )


# batch upsert

BATCH = 100
for i in range(0, len(records), BATCH):
    attempt = 0
    while True:
        try:
            index.upsert_records(namespace="__default__", records=records[i:i+BATCH])
            break
        except exceptions.ServiceUnavailableException as e:
            attempt += 1
            backoff = 2 ** attempt
            logging.warning(f"Retry in {backoff}s – {e}")
            time.sleep(backoff)
logging.info(f"Uploaded {len(records)} bugs")


# check total records

stats = index.describe_index_stats()

logging.info(f"Total records: {stats["total_vector_count"]}")