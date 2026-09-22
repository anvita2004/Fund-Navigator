import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import app_graph

with open("evaluation/test_questions.json", "r") as f:
    test_cases = json.load(f)

correct = 0
results = []

for case in test_cases:
    output = app_graph.invoke({"query": case["question"], "intent": None, "result": None})
    predicted_intent = output["intent"]
    is_correct = case["expected_intent"] in predicted_intent
    correct += int(is_correct)

    results.append({
        "question": case["question"],
        "expected": case["expected_intent"],
        "predicted": predicted_intent,
        "correct": is_correct,
    })

    print(f"[{'PASS' if is_correct else 'FAIL'}] {case['question']}")
    print(f"   expected: {case['expected_intent']}, got: {predicted_intent}\n")

accuracy = correct / len(test_cases)
print(f"\nIntent routing accuracy: {accuracy * 100:.1f}% ({correct}/{len(test_cases)})")

with open("evaluation/results.json", "w") as f:
    json.dump({"accuracy": accuracy, "details": results}, f, indent=2)