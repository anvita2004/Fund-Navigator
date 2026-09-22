import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("data/funds.json", "r") as f:
    funds = json.load(f)

# This downloads a small free embedding model the first time you run it (~90MB)
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="funds")

documents = []
metadatas = []
ids = []

for i, fund in enumerate(funds):
    text = (
        f"{fund['fund_name']} is a {fund['category']} fund with {fund['risk_level']} risk. "
        f"Expense ratio: {fund['expense_ratio']}%. 3-year return: {fund['three_year_return']}%. "
        f"{fund['description']}"
    )
    documents.append(text)
    metadatas.append(fund)
    ids.append(f"fund_{i}")

embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)

print(f"Successfully indexed {len(documents)} funds into ChromaDB.")