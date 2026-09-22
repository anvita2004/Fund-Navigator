# FundNavigator

A small agent that answers questions about mutual funds — not by matching keywords, but by figuring out *what kind* of question you're actually asking, pulling the right data, and giving back an answer it can prove is correctly structured.

I built this to go deeper into one specific problem I kept running into with LLM apps: most "RAG chatbots" just retrieve some text and let the model say whatever it wants in response. That's fine for a demo, but it falls apart the moment you need the output to plug into anything else — a UI, a database, another API call. So I wanted to build something where the model's output is *guaranteed* to be usable, not just plausible-sounding.

## How it works

Say you ask: *"Compare HDFC Balanced Advantage Fund and Axis Long Term Equity Fund."*

1. **Intent classification** — before doing anything else, the system decides what kind of question this is: a simple lookup, a comparison across funds, or a calculation (like "what would I pay in fees on a ₹50,000 investment"). This routing is handled by a [LangGraph](https://github.com/langchain-ai/langgraph) state machine, so the path a query takes is explicit and traceable, not buried inside one giant prompt.
2. **Retrieval** — relevant fund data is pulled from a local vector store ([ChromaDB](https://www.trychroma.com/)), built by embedding fund descriptions with `sentence-transformers`.
3. **Generation, but constrained** — the LLM writes the answer, but it's not free to return arbitrary text. Every response is validated against a [Pydantic](https://docs.pydantic.dev/) schema before it's accepted. If the model returns something malformed, the system catches it instead of quietly passing garbage downstream.
4. **Evaluation** — I didn't want to just claim this works; I wanted a number. There's a small labeled test set that runs through the full pipeline and reports how often the routing actually gets the intent right.

## Note on model choice

Runs entirely on Llama 3.2 via Ollama rather than a paid API — no data leaves the machine, no billing dependency. Trade-offs of a smaller local model (like occasional intent misclassification) are what the evaluation suite is designed to catch.

## Stack

Python, LangGraph, LangChain, ChromaDB, sentence-transformers, Pydantic, Ollama (Llama 3.2)

## Project layout

```
finrag/
├── data/                # sample fund dataset
├── ingestion/           # embeds and indexes fund data into ChromaDB
├── agent/
│   ├── schemas.py       # Pydantic models the LLM's output must satisfy
│   ├── nodes.py         # retrieval + generation logic per intent
│   └── graph.py         # the LangGraph routing definition
├── evaluation/          # labeled test set + accuracy scoring script
└── app.py               # command-line entry point
```

## Running it

```bash
# 1. Install Ollama and pull the model
ollama pull llama3.2

# 2. Set up the environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Build the vector index
python ingestion/build_vectorstore.py

# 4. Run it
python app.py

# 5. (Optional) See how well the routing actually performs
python evaluation/evaluate.py
```

## Sample run

```
Your question: If I invest ₹50,000 in SBI Magnum Low Duration Fund, what's the approximate annual expense?

[Detected intent: calculation]

Fund: SBI Magnum Low Duration Fund
--------------------------------------------------
Annual expense = 50,000 × 0.95% = ₹475

Risk Level: Low
Expense Ratio: 0.95%
```

## What I'd build next

If I kept going, the next step would be hybrid retrieval — combining keyword search (BM25) with the current embedding-based search and reranking the results, since pure dense retrieval isn't always the best choice for exact terms like fund names. I'd also want to replace the LLM-based intent router with a small trained classifier and compare the two head-to-head on accuracy and latency, rather than assuming the LLM approach is best by default.
