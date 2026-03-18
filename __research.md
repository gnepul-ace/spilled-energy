# Spilled Energy - Codebase Research Report

## 1. Project Overview

**Spilled Energy** is a Python library implementing a training-free method for detecting hallucinations in Large Language Models (LLMs). It was published at **ICLR 2026** (arXiv: 2602.18671) by Adrian Robert Minut, Hazem Dewidar, and Iacopo Masi from the OmnAI Lab. The project is licensed under Apache 2.0.

The core idea reinterprets the final softmax classifier of an LLM as an **Energy-Based Model (EBM)**, decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference time. This provides a principled, physics-inspired framework for measuring where "energy spills" during decoding — and these spills empirically correlate with factual errors, biases, and hallucinations.

---

## 2. Theoretical Foundation

### 2.1 The Core Insight: Energy Conservation Across Steps

The method proposes that a consistent language model should satisfy an energy conservation law: the energy implied by selecting a token `x_t` should match the energy of the system state once that token has been processed. Violations of this equality are the "spilled energy."

### 2.2 The Three Metrics

The library computes three quantities at each token position:

#### A. Logit Energy `E(x_t)`
At step `t`, the model produces logits `f(x)` for all possible next tokens. The energy of the specific token `x_t` that was chosen is:

```
E(x_t) = -beta * f(x_{t-1}, ..., x_0)[Id(x_t)]
```

This is the negated logit value (scaled by inverse temperature `beta`) for the selected token — i.e., the energy "predicted" for token `x_t` before it is fully integrated into the context.

#### B. Marginal (Free) Energy `E_margin` / `F(x_{<t+1})`
After token `x_t` is appended to the context, the model computes new logits for the next step. The free energy of this new distribution over the vocabulary `V` is:

```
F(x_{<t+1}) = -LogSumExp_{v in V}(beta * Logit(v))
```

This is the negative log-partition function (log-sum-exp) of the logits at the new step. It represents the "stability" or normalization energy of the new state.

#### C. Spilled Energy `E_delta`
The spilled energy is the difference between these two quantities:

```
E_delta = E(x_t) - E_margin = -E_margin + E
```

Note: In the actual code (`energy.py:96`), the computation is `delta = -E_margin + E`, which means `delta = LogSumExp(logits) - beta * logit[chosen_token]`. Positive delta indicates the log-partition function exceeds the token's logit energy.

### 2.3 Interpretation

- **Low Spill (E_delta ≈ 0):** The model is confident and internally consistent. The logit value it assigned to `x_t` aligns with the stability of the state after picking it.
- **High Spill (E_delta >> 0):** The model is "surprised" or inconsistent. The state's energy after integrating the token does not match the prediction — a strong signal of hallucination.

### 2.4 Key Properties

- **Training-free:** No probe classifiers, activation ablations, or fine-tuning required.
- **Derived from output logits only:** Works as a post-hoc analysis on any autoregressive LLM.
- **Two complementary metrics:** Spilled energy (cross-step comparison) and marginalized energy (single-step measurable).
- **Generalizes across models:** Validated on LLaMA, Mistral, Gemma, and Qwen3 families, both pretrained and instruction-tuned.

---

## 3. Repository Structure

```
spilled-energy/
├── src/
│   ├── spilled_energy/          # Core library package
│   │   ├── __init__.py          # Package marker (exports nothing)
│   │   ├── energy.py            # Core energy computation functions
│   │   ├── model.py             # HuggingFace model loading utilities
│   │   ├── generation.py        # LLM text generation with logit capture
│   │   ├── extraction.py        # Exact answer extraction from long-form text
│   │   └── utils.py             # Plotting, metrics, batch logit computation
│   └── scripts/
│       ├── benchmark_methods.py # Full TriviaQA benchmark pipeline
│       └── test_measure_exact_answer.py  # Single-sample diagnostic script
├── notebooks/
│   ├── synth_maths.ipynb        # Synthetic math experiment (core paper experiment)
│   ├── measure_exact_answer.ipynb  # Step-by-step single-sample demo
│   └── benchmark_methods.ipynb  # Full benchmarking notebook
├── documentation/
│   ├── introduction.md          # Conceptual overview
│   ├── installation.md          # Setup guide
│   ├── usage.md                 # Usage walkthrough
│   ├── api_reference.md         # Module/function API docs
│   └── scripts.md               # Script documentation
├── media/
│   ├── fig1.png                 # Paper figure: energy visualization on text
│   └── tab1.png                 # Paper table: benchmark results on 9 datasets
├── pyproject.toml               # Project metadata and dependencies
├── .python-version              # Python 3.11
├── .env.example                 # HuggingFace token template
├── .gitignore
├── LICENSE.md                   # Apache 2.0
└── README.md                    # Project overview with badges
```

---

## 4. Core Library Modules (Deep Dive)

### 4.1 `energy.py` — The Heart of the Library

This module contains four functions implementing the spilled energy computation at different levels of abstraction:

#### `compute_softmax_denominator(logits, beta=1.0)`
Simple helper: computes `logsumexp(beta * logits)` along the vocabulary dimension. This is the log-partition function.

#### `compute_token_logit(logits, token_id)`
Simple helper: extracts the logit value for a specific token ID from a logit vector.

#### `spilled_energy(logits, ids, beta=1.0, prompt_length=0)` — Primary function
The main Python-list-based implementation. Takes nested lists of logits and token IDs.

**Algorithm:**
1. **Compute E:** For each sequence `j` and position `i > 0`: `E[j][i] = -beta * logits[j][i-1][ids[j][i]]`. Position 0 gets E=0 (BOS token). Note the index shift: `logits[i-1]` is used with `ids[i]`, because the logit distribution at step `i-1` predicts what token `i` will be.
2. **Compute E_margin:** For each position: `E_margin[j][i] = -logsumexp(beta * logits[j][i])`. This uses the logits at position `i` itself (the state after the token is integrated).
3. **Compute delta:** `delta[j][i] = -E_margin[j][i] + E[j][i]` = `logsumexp(logits[i]) - beta * logit_for_chosen_token_at_step(i-1)`.
4. **Prompt trimming:** If `prompt_length > 0`, strips the first `prompt_length` values from delta, E_margin, and E (so only the generated portion is analyzed).

**Returns:** Tuple of `(delta, E_margin, E)`, each being a list of lists.

#### `spilled_energy_torch(logits, ids, beta=1.0, pad_token_id=-100)` — Batched tensor version
A more efficient PyTorch-native version operating on tensors of shape `[batch, seq_len, vocab_size]`. Uses `torch.take_along_dim` for gathering token-specific logits and handles padding token masking.

#### `spilled_energy_last_token(logits, ids, beta=1.0)` — Incremental version
Computes spilled energy for only the last token position. Useful for streaming/incremental decoding scenarios.

### 4.2 `generation.py` — Answer Generation

Single function `generate_answer()` that wraps HuggingFace's `model.generate()` with:
- `output_scores=True` — critical for capturing per-step logits needed for energy computation
- `return_dict_in_generate=True` — returns structured output
- Returns a dict with `text`, `sequences` (token IDs), and `scores` (tuple of logit tensors per generation step)
- Supports greedy and sampling-based decoding

### 4.3 `extraction.py` — Exact Answer Extraction

The `extract_exact_answer()` function uses the LLM itself (or another model) to extract a short/exact answer from a long-form generation. This is important because spilled energy is most meaningful when computed on the specific answer tokens rather than the full verbose response.

**Default prompt template:** Uses a 2-shot few-shot prompt with:
- Example 1: "My Fair Lady" extracted from a verbose answer about the musical
- Example 2: "NO ANSWER" when the model's answer doesn't actually answer the question

**Post-processing:** Strips whitespace, takes only the first line, removes trailing periods, and checks for "NO ANSWER" responses.

### 4.4 `model.py` — Model Loading

`load_model_and_tokenizer()` provides a convenience wrapper around HuggingFace's `AutoModelForCausalLM` and `AutoTokenizer` with:
- Automatic device detection (CUDA/CPU)
- Dtype mapping (float16/float32/bfloat16, with float32 forced on CPU)
- Support for `BitsAndBytesConfig` quantization
- Automatic pad token configuration (set to EOS if missing)
- Automatic eval mode

### 4.5 `utils.py` — Utilities (Large Module)

This is the largest module with multiple categories of functionality:

#### Reproducibility
- `set_seed()`: Sets seeds across random, numpy, torch, CUDA, and HuggingFace, with optional deterministic CUDA mode.

#### Answer Processing
- `clean_answer()`: Removes text after common stop patterns (e.g., `\nQ:`, `\nHuman:`, etc.)

#### Metrics Aggregation
- `compute_metrics()`: Aggregates per-sample results into mean, std, median, min, max of spilled energy values.

#### Visualization (6 functions)
- `plot_tokens()`: Color-codes tokens using RdYlGn colormap based on energy values, rendered as HTML. Supports annotations at sentence boundaries.
- `plot_examples()`: Convenience wrapper plotting up to 3 examples.
- `plot_histogram()`: Histograms comparing correct vs. incorrect energy distributions with configurable pooling (min, max, mean, etc.).
- `plot_PR_curve()` / `plot_ROC_curve()`: Single-method precision-recall and ROC curves.
- `plot_multiple_PR_curves()` / `plot_multiple_ROC_curves()`: Multi-method comparison curves with AUROC/AP annotations.

#### Batch Logit Computation
- `compute_logits()`: Runs a forward pass through the model and returns logits + input IDs as numpy arrays.
- `compute_logits_batch()`: Batched version with configurable batch size and tqdm progress bar.
- `remove_pad_tokens_from_logits()`: Trims padding tokens from logits and IDs (supports both left and right padding).

#### Probability Computation
- `compute_joint_p()`: Computes the total log joint probability of each sequence.
- `compute_joint_p_sequence()`: Computes the running log joint probability at each subsequence position.

#### HellaSwag Formatting
- `format_prompt_hellaswag()` / `format_question_hellaswag()`: Formatting utilities for the HellaSwag benchmark (multiple choice sentence completion).

---

## 5. Scripts and Notebooks

### 5.1 `test_measure_exact_answer.py` — Diagnostic Script

End-to-end pipeline verification on a single TriviaQA sample:
1. Loads Meta-Llama-3-8B (falls back to OPT-125M if unavailable)
2. Generates a long-form answer
3. Extracts the exact short answer
4. Maps the extracted answer to token indices in the generated sequence
5. Computes spilled energy on both the exact answer tokens and the full generation
6. Prints Mean/Max/Min/Sum statistics for delta, E, and E_margin

### 5.2 `benchmark_methods.py` — Full Benchmark Script

Systematic evaluation pipeline:
1. Processes 75 TriviaQA samples (25 validation + 50 test)
2. For each sample: generate → extract → check correctness (exact match against aliases) → locate tokens → compute metrics
3. On validation set: finds optimal F1 threshold for each metric/strategy combination
4. On test set: computes AUROC for hallucination detection
5. Reports a grid of results: 3 metrics (delta, E, E_margin) × 4 strategies (mean, max, min, sum) = 12 configurations
6. Highlights the best-performing method

### 5.3 `synth_maths.ipynb` — Synthetic Mathematics Experiment (Key Paper Experiment)

The most substantial notebook. Tests the method on **synthetic arithmetic** where ground truth is known exactly:

**Dataset generation:**
- Creates 1000 math addition problems (`a + b = x`) with known correct answers
- Creates perturbed versions with controlled error magnitudes
- Three scales of operand magnitude: ~10^8, ~10^11, ~10^13
- Multiple error ranges: [1,10], [10,100], [100,100000] — representing "hard" to "easy" detection difficulty

**Analysis pipeline:**
- Computes logits in batches using 4-bit NF4 quantization (BitsAndBytes)
- Caches computed logits to disk as pickle files for re-use
- Removes padding tokens before analysis
- Computes three metrics on the answer portion only (strips question tokens)
- Also computes joint probability sequences as a baseline comparison

**Visualizations produced:**
- Histograms of spilled energy distributions (correct vs. incorrect)
- Box plots comparing energy distributions
- Joint probability histograms (log-likelihood baseline)
- Energy and marginalized energy histograms
- ROC curves comparing all three methods (spilled energy, energy, marginalized energy) with AUROC scores
- Combined ROC plots with different line styles per error range and logarithmic FPR axis

### 5.4 `measure_exact_answer.ipynb` — Interactive Single-Sample Demo

Interactive notebook version of the test script. Adds a simple threshold-based hallucination classifier at the end (threshold = 0.5 on mean spilled energy).

### 5.5 `benchmark_methods.ipynb` — Interactive Benchmark

Notebook version of the benchmark script. Contains actual execution results showing:
- Val Accuracy: 48%, Test Accuracy: 46%
- Best performing method: `spilled (sum)` with AUROC 0.649
- Spilled energy with `max` and `sum` strategies outperform raw energy and marginalized energy

---

## 6. Dependencies and Build System

### Python Version
- Requires Python >= 3.11 (pinned to 3.11 in `.python-version`)

### Key Dependencies (from `pyproject.toml`)
| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.8.0 | Core tensor computation |
| `transformers` | >= 5.0.0 | HuggingFace model loading/generation |
| `datasets` | >= 4.0.0 | TriviaQA and other datasets |
| `accelerate` | >= 1.1.0 | Model distribution/device mapping |
| `bitsandbytes` | >= 0.46.1 | 4-bit/8-bit quantization |
| `scikit-learn` | >= 1.8.0 | AUROC, precision-recall metrics |
| `matplotlib` | >= 3.10.8 | Plotting |
| `seaborn` | >= 0.13.2 | Statistical visualization |
| `dotenv` | >= 0.9.0 | Environment variable loading |

### Build System
- Uses `setuptools` with `pyproject.toml`
- Source layout: packages found under `src/`
- Recommended installation via `uv sync`

---

## 7. Design Patterns and Architectural Observations

### 7.1 Dual Implementation Strategy
The energy computation exists in three variants:
- **Python-list version** (`spilled_energy`): Flexible, works with variable-length sequences, used in all scripts/notebooks
- **PyTorch tensor version** (`spilled_energy_torch`): Batched, GPU-efficient, handles padding masks
- **Incremental version** (`spilled_energy_last_token`): For streaming/online scenarios

In practice, only the Python-list version is used in the scripts and notebooks. The torch version exists but is not exercised by any of the provided scripts.

### 7.2 Self-Referential Extraction
The library uses the **same LLM** for both generation and extraction (extracting the exact answer from the verbose generation). This is efficient but means extraction quality depends on the model's instruction-following ability.

### 7.3 Token Alignment Challenge
A recurring pattern in both scripts is the need to **align extracted answer text back to token positions** in the generated sequence. This involves:
1. Finding the substring in the generated text
2. Re-tokenizing with `return_offsets_mapping=True`
3. Mapping character offsets to token indices
4. Bounds checking and fallback to full-sequence analysis

This is fragile — the code includes warnings about potential misalignment when re-tokenization produces different results than the original generation.

### 7.4 Aggregation Strategies
The library evaluates four aggregation strategies for reducing per-token energy values to a single sequence-level score: **mean**, **max**, **min**, and **sum**. The documentation notes that `min` or `max` often work best depending on the metric, and the benchmark results confirm `sum` and `max` as strong for spilled energy.

### 7.5 Evaluation Framework
The evaluation assumes a binary classification setup:
- **Positive class:** Hallucination (incorrect answer)
- **Negative class:** Correct answer
- Threshold found by maximizing F1 on validation set
- Performance reported as AUROC on held-out test set

---

## 8. Notable Specificities

### 8.1 The beta Parameter
All energy functions accept a `beta` (inverse temperature) parameter defaulting to 1.0. This allows scaling the logits before energy computation. In all provided scripts and notebooks, `beta=1.0` is always used.

### 8.2 Index Shift in Energy Computation
A critical detail: when computing `E[j][i]`, the code uses `logits[j][i-1][ids[j][i]]` — the logit distribution from the **previous** step is indexed by the **current** token ID. This captures the energy the model assigned to the chosen token **before** it was incorporated into the context. Position 0 (BOS) always gets `E=0`.

### 8.3 Sign Convention
- `E` values are typically negative (negated logit of chosen token)
- `E_margin` values are always negative (negated log-sum-exp, which is always positive)
- `delta = -E_margin + E`: positive delta means the partition function energy exceeds the token energy

### 8.4 Quantization Support
The synthetic math notebook uses **NF4 4-bit quantization** via BitsAndBytes for running experiments with large models on consumer GPUs. The model loading utility also supports this via `quantization_config` parameter.

### 8.5 Paper Claims vs. Codebase
The paper abstract mentions evaluation on **9 benchmarks** across LLaMA, Mistral, and Gemma models. The codebase itself only includes TriviaQA benchmarking and synthetic math. The README notes that the full paper experiments used code from the [LLMsKnow repository](https://github.com/technion-cs-nlp/LLMsKnow) by Orgad et al.

### 8.6 HellaSwag Support
The utils module includes HellaSwag formatting functions (`format_prompt_hellaswag`, `format_question_hellaswag`), suggesting the method was also tested on multiple-choice sentence completion tasks, though no HellaSwag-specific notebook or script is provided.

### 8.7 Joint Probability as Baseline
The synthetic math notebook computes joint log-probability (`compute_joint_p_sequence`) as a comparison baseline, allowing direct comparison between spilled energy and traditional sequence probability as hallucination detection signals.

---

## 9. Workflow Summary

The typical usage flow is:

```
1. Load model + tokenizer (model.py or direct HuggingFace)
       ↓
2. Generate answer with logits captured (generation.py)
       ↓
3. Extract exact answer from verbose text (extraction.py)
       ↓
4. Map extracted answer to token positions (manual alignment)
       ↓
5. Compute spilled energy on answer tokens (energy.py)
       ↓
6. Aggregate per-token values (mean/max/min/sum)
       ↓
7. Threshold to classify as hallucination or reliable
```

---

## 10. Strengths and Limitations Observed

### Strengths
- **Elegant theoretical grounding** in energy-based models
- **Zero training overhead** — works with any autoregressive LLM out of the box
- **Clean, focused codebase** with well-separated concerns
- **Multiple implementation variants** (list, tensor, incremental) for different use cases
- **Rich visualization** support for qualitative analysis
- **Reproducible** with seed-setting utilities

### Limitations
- **Token alignment fragility** — re-tokenization may not perfectly match generation tokenization
- **Single-model extraction** — using the same model for extraction as for generation may not be robust for all models
- **No unit tests** — the "test" script is actually a diagnostic/demo, not automated tests
- **`__init__.py` exports nothing** — `__all__ = []` means users must import from submodules directly
- **Torch version unused** — `spilled_energy_torch` is implemented but never called in any script or notebook
- **Small sample sizes** in demos — benchmarks use 25+50=75 samples, which is acknowledged as a toy demonstration
