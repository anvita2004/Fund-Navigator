from agent.graph import app_graph
import json


def print_human_readable(result: dict):
    """Formats the structured answer into plain, readable text."""
    if "funds_compared" in result:
        # This is a ComparisonResult
        print(f"\nComparing: {', '.join(result['funds_compared'])}")
        print("-" * 50)
        for metric in result["comparison"]:
            print(f"{metric['metric_name'].replace('_', ' ').title()}:")
            for fund, value in metric["values"].items():
                print(f"   {fund}: {value}")
        print("-" * 50)
        print(f"Conclusion: {result['conclusion']}")
    else:
        # This is a FundAnswer (lookup or calculation)
        print(f"\nFund: {result['fund_name']}")
        print("-" * 50)
        print(result["answer_summary"])
        if result.get("risk_level"):
            print(f"\nRisk Level: {result['risk_level']}")
        if result.get("expense_ratio") is not None:
            print(f"Expense Ratio: {result['expense_ratio']}%")


def ask(query: str, show_raw_json: bool = False):
    result = app_graph.invoke({"query": query, "intent": None, "result": None})

    print(f"\n[Detected intent: {result['intent']}]")
    print_human_readable(result["result"])

    if show_raw_json:
        print("\n(Raw structured output:)")
        print(json.dumps(result["result"], indent=2))


if __name__ == "__main__":
    print("FundNavigator — ask a question about mutual funds (type 'quit' to exit)")
    print("(Tip: type 'quit' to exit, or add ' --json' after your question to also see raw structured output)\n")

    while True:
        query = input("Your question: ")
        if query.lower() == "quit":
            break

        show_json = query.strip().endswith("--json")
        clean_query = query.replace("--json", "").strip()

        ask(clean_query, show_raw_json=show_json)