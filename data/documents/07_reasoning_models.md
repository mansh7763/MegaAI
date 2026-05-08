# Reasoning Models

## What Are Reasoning Models?
Reasoning models are LLMs trained to "think before answering" — they generate an extended chain of reasoning tokens (a scratchpad) before producing a final answer. This allows them to solve complex multi-step problems that one-shot generation fails on.

## Chain-of-Thought (CoT) Prompting — (Wei et al., 2022)
The original technique: prompt the model with few-shot examples that include step-by-step reasoning.
- "Let's think step by step" elicits reasoning even zero-shot (Kojima et al., 2022)
- CoT improves performance on arithmetic, commonsense, and symbolic reasoning
- **Emergent**: CoT only helps for models ≥100B parameters in the original work; smaller models benefit with fine-tuning

## OpenAI o1 (2024)
o1 uses **reinforcement learning on chain-of-thought reasoning**:
- Model generates a hidden "thinking" trace (not shown to user by default)
- Trained via RLHF/GRPO to produce accurate reasoning chains that lead to correct answers
- The thinking trace can be thousands of tokens for hard problems
- **Test-time compute scaling**: more thinking tokens = better answers (unlike standard models where more output tokens don't help reasoning quality)
- Excels at: competition math, PhD-level science, competitive programming

## OpenAI o3 (2025)
- Extends o1 with significantly longer thinking traces
- Achieves near-human performance on ARC-AGI (a test designed to be hard for LLMs)
- Uses a search-like approach during thinking (not just linear chain-of-thought)

## DeepSeek-R1 (2025)
Open-source reasoning model from DeepSeek:
- Uses **GRPO (Group Relative Policy Optimisation)** for RL training
- Training process:
  1. Cold-start: SFT on long CoT examples
  2. RL: reward model based on answer correctness and format
  3. Rejection sampling: filter good reasoning traces
  4. Final SFT + RL
- Achieves performance comparable to o1 on math and coding benchmarks
- Fully open weights and training recipe

## Tree of Thought (ToT) — (Yao et al., 2023)
Instead of linear CoT, explore a tree of reasoning paths:
- Generate multiple candidate reasoning steps
- Evaluate each step (via LLM self-evaluation or external verifier)
- Backtrack and try alternative paths
- Best path wins
- Significant improvement on planning and puzzle tasks

## Process Reward Models (PRM)
Train a reward model to score individual reasoning steps (not just final answers):
- Provides denser training signal than outcome-based rewards
- Allows filtering training data to keep only correct reasoning traces
- Used in: OpenAI's o1 training (speculated), DeepSeek-R1

## Key Characteristics of Reasoning Models

| Property | Standard LLM | Reasoning Model |
|----------|-------------|-----------------|
| Inference compute | Fixed | Variable (scales with problem difficulty) |
| Reasoning visibility | Implicit | Explicit thinking trace |
| Math/logic benchmark | Moderate | State-of-the-art |
| Latency | Low | High (long thinking) |
| Training | SFT + RLHF | SFT + RL on reasoning |

## Reasoning Failures
- **Overthinking**: generating excessive tokens without improvement
- **Reward hacking**: learning to produce plausible-sounding reasoning that leads to wrong answers
- **Hallucinated reasoning**: incorrect premises in the thinking chain that nonetheless produce correct final answers by accident
