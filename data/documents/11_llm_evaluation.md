# LLM Evaluation

## Why Evaluation Is Hard
LLMs produce open-ended text — there's rarely a single correct answer. Evaluation must balance:
- **Correctness**: does the answer reflect the facts?
- **Fluency**: is the text well-formed?
- **Faithfulness**: does it stay grounded in sources (for RAG)?
- **Helpfulness**: does it address the user's actual need?

## Automatic Metrics

### Reference-based
- **BLEU**: n-gram precision vs reference. Good for translation. Poor for open-ended generation.
- **ROUGE-L**: longest common subsequence. Good for summarisation.
- **BERTScore**: contextual embedding similarity between generated and reference texts. Better semantic alignment than BLEU.

### Reference-free (LLM-as-Judge)
- Use a capable LLM (GPT-4, Claude) to score outputs on dimensions like accuracy, relevance, safety
- **G-Eval** (Liu et al., 2023): structured prompt asking LLM to score on 0-1 scale with CoT
- **MT-Bench**: 8 categories of multi-turn conversations, GPT-4 judged
- **Prometheus**: open-source judge model trained on human preference data

## Hallucination Detection
- **FactScore** (Min et al., 2023): decomposes answer into atomic claims, verifies each against a knowledge source
- **SelfCheckGPT**: samples multiple responses, checks consistency — inconsistent facts are likely hallucinated
- **RAGAs**: end-to-end RAG evaluation with metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall

## Adversarial Evaluation
- **TruthfulQA**: questions humans answer incorrectly due to misconceptions — tests if LLMs parrot falsehoods
- **AdvGLUE**: adversarially constructed NLI/sentiment examples
- **Prompt injection benchmarks**: test whether safety guidelines can be overridden by malicious instructions

## Multi-Dimensional Scoring (This System)
This system implements custom scoring on 6 dimensions:

| Dimension | What It Measures |
|-----------|-----------------|
| answer_correctness | Keyword match vs ground truth |
| citation_accuracy | Whether cited chunks support claims |
| contradiction_resolution | Whether critique flags were resolved |
| tool_efficiency | Penalise unnecessary tool calls |
| budget_compliance | Policy violation rate |
| critique_agreement | Critique agent agreement with final output |

## Eval Best Practices
1. **Never use eval data for training** (data leakage)
2. **Include adversarial cases** not just clean inputs
3. **Score dimensions separately** — overall score hides important failures
4. **Store raw outputs** not just scores — enables regression testing
5. **Reproducibility**: same input must produce diffable output across runs
