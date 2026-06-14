"""
MappingMind — RAGAS Evaluation
================================
CONCEPTS:
  - Faithfulness: answer grounded in context (no hallucination)
  - Answer Relevancy: answer addresses the question
  - Context Precision: retrieved chunks are relevant
  - Context Recall: all relevant chunks retrieved

WHY THIS MATTERS:
  "How do you know your RAG works?" 
  → Most candidates: "I tested it manually"
  → You: "Faithfulness 0.87, relevancy 0.91, measured on every PR"
  That answer wins offers.
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
from core.retrieval import retrieve
from core.generation import build_context, chain

load_dotenv()

# ── TEST DATASET ───────────────────────────────────────────────────
# Ground truth questions + expected answers
# Production: maintain 20-50 test cases, update when data changes

TEST_CASES = [
    {
        "question": "How should we map customer_account_balance from CoreBanking?",
        "ground_truth": "Use CAST(field/100 AS DECIMAL(18,2)) to normalize from minor currency units to USD. Confirmed by ADR-001."
    },
    {
        "question": "What is the standard transformation for date fields?",
        "ground_truth": "Convert all date fields to ISO 8601 format using TO_DATE(). For YYYYMM sources use LAST_DAY(). Defined in ADR-002."
    },
    {
        "question": "How should we handle string normalization for ID fields?",
        "ground_truth": "Apply TRIM(UPPER()) for ID fields. Use INITCAP(TRIM()) for description fields. Defined in ADR-003."
    },
    {
        "question": "What mapping patterns should we avoid?",
        "ground_truth": "Avoid implicit type casting, NULL assumption for numeric fields, and hardcoded CASE WHEN values. Documented in ADR-004."
    },
    {
        "question": "How was interest rate mapped from LoanSystem?",
        "ground_truth": "Use ROUND(int_rate * 100, 4) to convert decimal (0.0525) to percentage (5.25)."
    }
]


# ── RUN EVAL ───────────────────────────────────────────────────────
def run_evaluation() -> dict:
    print("\n🔬 Running RAGAS Evaluation")
    print("=" * 40)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, test in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {test['question'][:60]}...")

        try:
            # Retrieve
            results = retrieve(test["question"], top_k=5)
            context = build_context(results)

            # Generate
            answer = chain.invoke({
                "context": context,
                "question": test["question"]
            })

            questions.append(test["question"])
            answers.append(answer)
            contexts.append([doc.page_content for doc, _ in results])
            ground_truths.append(test["ground_truth"])

            print(f"   ✅ Generated answer: {answer[:80]}...")
            time.sleep(3)  # respect free tier rate limits

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

    if not questions:
        print("❌ No results to evaluate")
        return {}

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Run RAGAS
    print(f"\n📊 Running RAGAS metrics on {len(questions)} questions...")
    
    try:
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        results = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "faithfulness": round(float(scores["faithfulness"]), 3),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 3),
        }

        print(f"\n✅ RAGAS Results:")
        print(f"   Faithfulness:      {results['faithfulness']}")
        print(f"   Answer Relevancy:  {results['answer_relevancy']}")

        # Save results
        os.makedirs("data/eval_results", exist_ok=True)
        fname = f"data/eval_results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved to {fname}")

        return results

    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return {}


if __name__ == "__main__":
    results = run_evaluation()