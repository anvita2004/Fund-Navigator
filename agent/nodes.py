import json
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from agent.schemas import FundAnswer, ComparisonResult, ComparisonMetric

MODEL_NAME = "llama3.2"  # change to "phi3:mini" here if you used the smaller model

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="funds")


def ask_ollama(prompt: str) -> str:
    """Sends a prompt to your local free AI model and returns its reply."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def classify_intent(query: str) -> str:
    prompt = f"""Classify this question about mutual funds into exactly one word:
"lookup" (asking about one fund), "comparison" (comparing two or more funds),
or "calculation" (asking to compute a number, like investment cost).

Question: {query}

Reply with only one word: lookup, comparison, or calculation. No explanation."""
    result = ask_ollama(prompt).lower()
    # Keep only the first matching keyword in case the model adds extra words
    for keyword in ["lookup", "comparison", "calculation"]:
        if keyword in result:
            return keyword
    return "lookup"  # safe default


def retrieve_funds(query: str, top_k: int = 3):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["metadatas"][0], results["documents"][0]


def answer_lookup(query: str) -> FundAnswer:
    metadatas, documents = retrieve_funds(query, top_k=1)
    context = documents[0]
    fund_data = metadatas[0]

    prompt = f"""Using ONLY this information, answer the question in 2-3 short sentences.

Fund information: {context}

Question: {query}"""

    answer = ask_ollama(prompt)

    return FundAnswer(
        fund_name=fund_data["fund_name"],
        answer_summary=answer,
        risk_level=fund_data.get("risk_level"),
        expense_ratio=fund_data.get("expense_ratio"),
        sources=[fund_data["fund_name"]],
    )


def answer_comparison(query: str) -> ComparisonResult:
    metadatas, documents = retrieve_funds(query, top_k=3)
    context = "\n".join(documents)

    prompt = f"""Using ONLY this information about multiple funds, answer the comparison question.

Fund information:
{context}

Question: {query}

Give a short comparison (3-4 sentences) and one final sentence stating which fund best fits the question."""

    conclusion = ask_ollama(prompt)
    fund_names = [m["fund_name"] for m in metadatas]

    return ComparisonResult(
        funds_compared=fund_names,
        comparison=[
            ComparisonMetric(
                metric_name="expense_ratio",
                values={m["fund_name"]: f"{m['expense_ratio']}%" for m in metadatas},
            )
        ],
        conclusion=conclusion,
        sources=fund_names,
    )


def answer_calculation(query: str) -> FundAnswer:
    metadatas, documents = retrieve_funds(query, top_k=1)
    context = documents[0]
    fund_data = metadatas[0]

    prompt = f"""Using this fund information, perform the calculation asked for.
Show your reasoning briefly, then state the final number clearly.

Fund information: {context}

Question: {query}"""

    answer = ask_ollama(prompt)

    return FundAnswer(
        fund_name=fund_data["fund_name"],
        answer_summary=answer,
        risk_level=fund_data.get("risk_level"),
        expense_ratio=fund_data.get("expense_ratio"),
        sources=[fund_data["fund_name"]],
    )