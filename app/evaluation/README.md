# RAGAS Evaluation

This package evaluates RAG output with the dataset schema:

```json
{
  "question": "...",
  "ground_truth": "...",
  "contexts": ["..."],
  "answer": "..."
}
```

Run:

```bash
python scripts/run_ragas_eval.py
```

The runner writes per-question scores to the `evaluation_scores` table and exports
`evals/reports/ragas_report.csv`. It also writes `evals/reports/ragas_summary.json`
with average scores, worst questions, and best questions.

## Dataset Sources

The dataset builder combines:

- Manual benchmark questions for Apple, Amazon, and Microsoft.
- Synthetic questions stored on ingested chunks in `chunks.synthetic_questions`.

Manual benchmarks define expected `ground_truth` answers and metadata filters. Synthetic
examples use each chunk's generated question, chunk summary or content as ground truth, and
the pipeline's generated answer and retrieved contexts.

## Provider-Agnostic RAGAS Execution

`RagasRunner` accepts optional `llm`, `embeddings`, and metric objects. This keeps the
evaluation provider-agnostic: callers can pass any RAGAS-compatible LLM and embedding
adapter instead of relying on a hard-coded vendor.

If RAGAS cannot run locally, for example because provider adapters are missing or a local
dependency is broken, the runner can fall back to deterministic lexical approximations.
Disable that behavior with:

```bash
python scripts/run_ragas_eval.py --no-fallback
```

## Metrics

### Faithfulness

Faithfulness measures whether claims in the answer are supported by the retrieved contexts.

Mathematically:

```text
faithfulness = supported_answer_claims / total_answer_claims
```

Example: if an answer makes four factual claims and three can be inferred from the supplied
contexts, faithfulness is `3 / 4 = 0.75`.

High faithfulness means the answer is grounded. Low faithfulness means the answer may be
hallucinating or using information outside the retrieved evidence.

### Answer Relevancy

Answer relevancy measures how directly the answer addresses the question.

Mathematically, RAGAS estimates whether the answer can generate semantically similar
questions to the original question:

```text
answer_relevancy = similarity(original_question, questions_generated_from_answer)
```

Example: for "What supply chain risks does Apple disclose?", an answer about supplier
availability and logistics disruptions should score high. An answer about unrelated revenue
growth should score low even if it is factually correct.

### Context Precision

Context precision measures whether the retrieved contexts ranked near the top are relevant
to the question.

Mathematically:

```text
context_precision = average precision over retrieved contexts
```

For a ranked list, precision is recomputed each time a relevant context appears:

```text
precision@k = relevant_contexts_in_top_k / k
context_precision = mean(precision@k for each relevant context position)
```

Example: if the top three contexts are `[relevant, irrelevant, relevant]`, the relevant
positions are 1 and 3, so the score is `(1/1 + 2/3) / 2 = 0.8333`.

High context precision means the retriever places useful evidence early. Low precision means
the answer generator must filter through noise.

### Context Recall

Context recall measures whether the retrieved contexts contain the information needed to
answer the question according to the ground truth.

Mathematically:

```text
context_recall = ground_truth_claims_supported_by_contexts / total_ground_truth_claims
```

Example: if the ground truth contains five important claims and the retrieved contexts
support four of them, context recall is `4 / 5 = 0.80`.

High context recall means retrieval found enough evidence. Low recall means the retriever
missed required facts, even if the answer itself sounds plausible.
