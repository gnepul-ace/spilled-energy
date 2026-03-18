# Implementation Plan: MI-Reframed Spilled Energy

## Overview

This plan implements the four new features described in the theoretical paper "MI-Reframed Spilled Energy," which unifies EvoRate (NeurIPS 2024) and Spilled Energy (ICLR 2026) via information-theoretic decomposition. The features are:

1. **Conditional Entropy / MI Proxy** — `H_Q(X_{i+1} | x_{i:1})` and `MI_proxy(i)`
2. **Excess Surprise** — `ε(x_i) = s(x_i) - H_Q(X_i | x_{i-1:1})`
3. **MI-Calibrated Spilled Energy** — `ΔE_cal = ΔE / (1 + α·H_Q)`
4. **Diagnostic Taxonomy** — 4-quadrant classification: HALLUCINATION / UNCERTAIN / CORRECT / LUCKY GUESS

All quantities are computed from logits alone, requiring zero training.

---

## File Change Map

```
src/spilled_energy/
├── energy.py          ← MODIFY: add 5 new functions
├── mi.py              ← NEW: MI proxy, excess surprise, calibration, taxonomy
├── __init__.py        ← MODIFY: export new public API
src/scripts/
├── benchmark_mi.py    ← NEW: benchmark comparing original vs MI-reframed metrics
notebooks/
├── mi_diagnostic.ipynb ← NEW: interactive taxonomy visualization
documentation/
├── mi_theory.md        ← NEW: theory & usage docs for the new features
├── api_reference.md    ← MODIFY: add new module/function docs
```

---

## Step 1: Core Computations in `energy.py`

Add two foundational functions to `energy.py` that compute **conditional entropy** and **pointwise surprise** from logits. These are building blocks used by everything else, and they belong in `energy.py` because they operate at the same level of abstraction as the existing energy functions (raw logit tensors in, scalar values out).

### 1.1 `compute_conditional_entropy`

This implements Definition 5 from the paper (Eq. 16):

```
H_Q(X_{i+1} | x_{i:1}) = -Σ_k Q(k | x_{i:1}) · log Q(k | x_{i:1})
```

where `Q(k | x_{i:1}) = softmax(θ(x_{i:1}))[k]`.

**Add to `src/spilled_energy/energy.py`:**

```python
def compute_conditional_entropy(
    logits: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """
    Compute conditional entropy H_Q(X_{i+1} | context) from logits.

    H_Q = -sum_k Q(k) * log Q(k)  where Q = softmax(beta * logits)

    This is the expected surprise of the model's own distribution -- how
    uncertain the model is about the next token given the current context.

    Args:
        logits: Tensor of shape (..., vocab_size) — raw logits at one or
                more positions.
        beta: Inverse temperature scaling factor (default 1.0).

    Returns:
        Tensor of shape (...) — conditional entropy at each position (nats).
        Values range from 0 (deterministic) to log(V) (uniform).
    """
    log_probs = torch.log_softmax(beta * logits, dim=-1)  # (..., V)
    probs = torch.exp(log_probs)                          # (..., V)
    entropy = -torch.sum(probs * log_probs, dim=-1)       # (...)
    return entropy
```

**Why this shape:** Accepts arbitrary leading dimensions so it works on single positions `(V,)`, full sequences `(seq_len, V)`, and batches `(batch, seq_len, V)` without reshaping.

### 1.2 `compute_surprise`

This implements Eq. 4 from the paper:

```
s(x_i) = -log p_θ(x_i | x_{i-1:1})
```

Which, via Eq. 12 (log-softmax decomposition), equals:

```
s(x_i) = -θ(x_{i-1:1})[id(x_i)] + logsumexp(θ(x_{i-1:1}))
       = E_ℓ(x_{i:1}) - E_m(x_{i-1:1})
```

This is the negative log-probability of the actually-chosen token — the standard per-token cross-entropy loss.

**Add to `src/spilled_energy/energy.py`:**

```python
def compute_surprise(
    logits: torch.Tensor, token_ids: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """
    Compute pointwise surprise s(x_i) = -log p(x_i | context).

    From the log-softmax decomposition (Lemma 1, Eq. 18):
        s(x_i) = E_logit(x_{i:1}) - E_margin(x_{i-1:1})
               = -logit[id(x_i)] + logsumexp(logits)

    Args:
        logits: Tensor of shape (..., vocab_size) — logits at each position.
                These are the logits from the step that *predicts* token x_i,
                i.e., logits at step i-1 in the autoregressive chain.
        token_ids: Tensor of shape (...) — the token IDs that were chosen.
        beta: Inverse temperature (default 1.0).

    Returns:
        Tensor of shape (...) — surprise values in nats.
        Always >= 0. Equals 0 only if the token has probability 1.
    """
    log_probs = torch.log_softmax(beta * logits, dim=-1)   # (..., V)
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)                                           # (...)
    return -token_log_probs
```

**Key difference from `E_logit`:** This returns `-log p(x_i)` (the full log-probability including normalization), not just `-logit[id(x_i)]` (the unnormalized energy). The relationship is `s(x_i) = E_logit - E_margin` (Lemma 1).

---

## Step 2: New Module `mi.py`

Create `src/spilled_energy/mi.py` — the main module for all MI-reframed features. This keeps the new theory cleanly separated from the original spilled energy code while reusing the building blocks from `energy.py`.

### 2.1 `mi_proxy` — MI Proxy (Proposition 3, Eq. 17)

```
MI_proxy(i) = max(0, log(V) - H_Q(X_{i+1} | x_{i:1}))
```

**Implementation in `src/spilled_energy/mi.py`:**

```python
"""
MI-Reframed Spilled Energy.

Implements information-theoretic extensions to spilled energy:
- MI proxy: training-free mutual information estimate from conditional entropy
- Excess surprise: pointwise model-error signal (operationalizes KL(P||Q))
- MI-calibrated spilled energy: false-positive suppression at low-MI positions
- Diagnostic taxonomy: 4-quadrant token classification

Reference: "MI-Reframed Spilled Energy" (unifying EvoRate + Spilled Energy)
"""

from typing import List, Optional, Tuple

import torch
import numpy as np

from spilled_energy.energy import (
    compute_conditional_entropy,
    compute_surprise,
    spilled_energy,
)


def mi_proxy(
    logits: torch.Tensor, beta: float = 1.0, vocab_size: Optional[int] = None
) -> torch.Tensor:
    """
    Compute MI proxy: a training-free lower bound on mutual information
    between the next token and the context.

    MI_proxy(i) = max(0, log(V) - H_Q(X_{i+1} | x_{i:1}))

    From Proposition 3 (Eq. 17): Since I(X;Y) = H(X) - H(X|Y) and
    H(X) <= log(V), we get I >= log(V) - H(X|Y) as an upper bound
    proxy. Clipped at 0 since MI is non-negative.

    Interpretation:
        - High MI_proxy: context strongly constrains the next token
          (model is confident, low entropy)
        - MI_proxy ≈ 0: distribution is near-uniform, context provides
          almost no information

    Args:
        logits: Tensor of shape (..., vocab_size).
        beta: Inverse temperature (default 1.0).
        vocab_size: Override vocabulary size V. If None, inferred from
                    logits.shape[-1].

    Returns:
        Tensor of shape (...) — MI proxy values in nats. Range [0, log V].
    """
    if vocab_size is None:
        vocab_size = logits.shape[-1]
    log_v = torch.tensor(float(np.log(vocab_size)), device=logits.device)
    h_cond = compute_conditional_entropy(logits, beta=beta)
    return torch.clamp(log_v - h_cond, min=0.0)
```

### 2.2 `excess_surprise` — Excess Surprise (Definition 6, Eq. 19)

```
ε(x_i) = s(x_i) - H_Q(X_i | x_{i-1:1})
       = (actual surprise) - (expected surprise)
```

From Proposition 5 (Eq. 22): `E_P[ε(X_i)] ≈ KL(P||Q)` when the model is good. This directly operationalizes the model-error component from EvoRate's Proposition 1.

```python
def excess_surprise(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute excess surprise: how much more surprising the chosen token is
    than the model expected.

    ε(x_i) = s(x_i) - H_Q(X_i | x_{i-1:1})

    From Proposition 5: E[ε] ≈ KL(P||Q), so excess surprise directly
    measures model error at each token position.

    Interpretation:
        - ε > 0: token is MORE surprising than expected (potential error)
        - ε ≈ 0: token matches the model's own uncertainty
        - ε < 0: token is LESS surprising than expected (very confident pick)

    Args:
        logits: Tensor of shape (..., vocab_size) — logits at each position.
                These are the logits from the step that predicts the token
                (step i-1 for token x_i).
        token_ids: Tensor of shape (...) — the chosen token IDs.
        beta: Inverse temperature (default 1.0).

    Returns:
        Tuple of (epsilon, surprise, h_cond):
        - epsilon: excess surprise values, shape (...)
        - surprise: pointwise surprise s(x_i), shape (...)
        - h_cond: conditional entropy H_Q at each position, shape (...)
    """
    s = compute_surprise(logits, token_ids, beta=beta)
    h_cond = compute_conditional_entropy(logits, beta=beta)
    epsilon = s - h_cond
    return epsilon, s, h_cond
```

### 2.3 `calibrated_spilled_energy` — MI-Calibrated ΔE (Definition 8, Eq. 26)

```
ΔE_cal(x_{i:1}) = ΔE(x_{i:1}) / (1 + α · H_Q(X_{i+1} | x_{i:1}))
```

This suppresses false positives at low-MI positions (punctuation, sentence boundaries) by a factor of up to `1 + α·log(V)` ≈ 12.8× for LLaMA-3's 128k vocabulary.

**Key subtlety:** The `H_Q` in the denominator uses logits from step `i` (the state *after* the token is integrated), which matches the `E_margin` time step. This is the right choice because we're asking "how uncertain is the model at the position where we're measuring the spill?"

```python
def calibrated_spilled_energy(
    logits: List[List[List[float]]],
    ids: List[List[int]],
    beta: float = 1.0,
    alpha: float = 1.0,
    prompt_length: int = 0,
) -> Tuple[list, list, list, list]:
    """
    Compute MI-calibrated spilled energy.

    ΔE_cal = ΔE / (1 + α * H_Q)

    From Proposition 7: at high-MI positions (H_Q ≈ 0), the denominator
    ≈ 1 and the raw signal is preserved. At low-MI positions (H_Q ≈ log V),
    the signal is suppressed by factor ~(1 + α·log V).

    This addresses the false-positive problem identified in Minut et al.
    §6 (Limitations): punctuation and sentence-initial tokens produce
    high |ΔE| not because of hallucination, but because the flat
    distribution amplifies cross-step discrepancies (Proposition 6).

    Args:
        logits: Nested list [batch][seq_len][vocab_size] of logit values.
        ids: Nested list [batch][seq_len] of token IDs.
        beta: Inverse temperature (default 1.0).
        alpha: Calibration strength (default 1.0). Higher alpha = more
               aggressive suppression at uncertain positions.
        prompt_length: Number of prompt tokens to exclude.

    Returns:
        Tuple of (delta_cal, delta_raw, h_cond, mi_proxy_vals):
        - delta_cal: calibrated spilled energy per token
        - delta_raw: uncalibrated spilled energy (for comparison)
        - h_cond: conditional entropy at each position
        - mi_proxy_vals: MI proxy at each position
    """
    # Get raw spilled energy using existing function
    delta_raw, E_margin, E = spilled_energy(
        logits=logits, ids=ids, beta=beta, prompt_length=prompt_length
    )

    # Compute H_Q at each position from logits
    # H_Q uses the logits at position i (same timestep as E_margin)
    h_cond_all = []
    mi_proxy_all = []
    vocab_size = len(logits[0][0]) if logits and logits[0] else 1
    log_v = float(np.log(vocab_size))

    for j in range(len(logits)):
        h_cond_j = []
        mi_proxy_j = []

        # Determine the range of positions matching delta after prompt trimming
        start = prompt_length if prompt_length > 0 and len(logits[j]) > prompt_length else 0
        end = len(logits[j])

        for i in range(start, end):
            logits_tensor = torch.tensor(logits[j][i], dtype=torch.float32)
            log_probs = torch.log_softmax(beta * logits_tensor, dim=-1)
            probs = torch.exp(log_probs)
            h = -torch.sum(probs * log_probs).item()
            h_cond_j.append(h)
            mi_proxy_j.append(max(0.0, log_v - h))

        h_cond_all.append(h_cond_j)
        mi_proxy_all.append(mi_proxy_j)

    # Calibrate: ΔE_cal = ΔE / (1 + α * H_Q)
    delta_cal = []
    for j in range(len(delta_raw)):
        delta_cal_j = []
        for i in range(len(delta_raw[j])):
            h = h_cond_all[j][i] if i < len(h_cond_all[j]) else 0.0
            denom = 1.0 + alpha * h
            raw = delta_raw[j][i]
            raw_val = raw.item() if hasattr(raw, 'item') else float(raw)
            delta_cal_j.append(raw_val / denom)
        delta_cal.append(delta_cal_j)

    return delta_cal, delta_raw, h_cond_all, mi_proxy_all
```

### 2.4 `classify_tokens` — Diagnostic Taxonomy (Section 7)

The 2D classification from the paper's table:

| | Low H_Q (high MI) | High H_Q (low MI) |
|---|---|---|
| **High \|ΔE\|** | HALLUCINATION | UNCERTAIN |
| **Low \|ΔE\|** | CORRECT | LUCKY GUESS |

```python
# Token classification labels
HALLUCINATION = "hallucination"
UNCERTAIN = "uncertain"
CORRECT = "correct"
LUCKY_GUESS = "lucky_guess"


def classify_tokens(
    delta: list,
    h_cond: list,
    tau_delta: float,
    tau_h: float,
) -> List[List[str]]:
    """
    Classify each token into the diagnostic taxonomy.

    Uses the 2D space (|ΔE|, H_Q) partitioned by thresholds (τ_δ, τ_h):

        |ΔE| > τ_δ  AND  H_Q < τ_h  →  HALLUCINATION
            Model had sufficient context info but failed.
            KL(P||Q) large, I(X;context) large.

        |ΔE| > τ_δ  AND  H_Q >= τ_h  →  UNCERTAIN
            Model is uncertain AND inconsistent. Even a perfect model
            would struggle — the irreducible loss H(X|context) is high.

        |ΔE| <= τ_δ  AND  H_Q < τ_h  →  CORRECT
            Confident and consistent. KL small, MI large.

        |ΔE| <= τ_δ  AND  H_Q >= τ_h  →  LUCKY GUESS
            Consistent but uninformed. Right by chance or default.

    Args:
        delta: Nested list [batch][seq_len] — spilled energy values
               (raw or calibrated).
        h_cond: Nested list [batch][seq_len] — conditional entropy values.
        tau_delta: Threshold on |ΔE| for high/low spill boundary.
        tau_h: Threshold on H_Q for high/low entropy boundary.

    Returns:
        Nested list [batch][seq_len] of classification strings.
    """
    labels = []
    for j in range(len(delta)):
        labels_j = []
        for i in range(len(delta[j])):
            d = abs(float(delta[j][i].item() if hasattr(delta[j][i], 'item') else delta[j][i]))
            h = float(h_cond[j][i])

            if d > tau_delta and h < tau_h:
                labels_j.append(HALLUCINATION)
            elif d > tau_delta and h >= tau_h:
                labels_j.append(UNCERTAIN)
            elif d <= tau_delta and h < tau_h:
                labels_j.append(CORRECT)
            else:
                labels_j.append(LUCKY_GUESS)

        labels.append(labels_j)
    return labels
```

### 2.5 `excess_surprise_sequence` — List-based Excess Surprise

A list-based variant matching the interface of `spilled_energy()` for consistency with the existing codebase patterns:

```python
def excess_surprise_sequence(
    logits: List[List[List[float]]],
    ids: List[List[int]],
    beta: float = 1.0,
    prompt_length: int = 0,
) -> Tuple[list, list, list]:
    """
    Compute excess surprise for sequences (list-based interface).

    ε(x_i) = s(x_i) - H_Q(X_i | x_{i-1:1})

    Both terms use logits from step i-1 (the prediction step).
    This contrasts with spilled energy, where E_margin uses step i.

    Args:
        logits: Nested list [batch][seq_len][vocab_size].
        ids: Nested list [batch][seq_len].
        beta: Inverse temperature (default 1.0).
        prompt_length: Number of prompt tokens to exclude.

    Returns:
        Tuple of (epsilon, surprise, h_cond):
        - epsilon: excess surprise per token [batch][seq_len]
        - surprise: pointwise surprise [batch][seq_len]
        - h_cond: conditional entropy [batch][seq_len]
    """
    epsilon_all = []
    surprise_all = []
    h_cond_all = []

    for j in range(len(logits)):
        eps_j = [0.0]   # position 0 (BOS): no prediction to evaluate
        s_j = [0.0]
        h_j = [0.0]

        for i in range(1, len(logits[j])):
            # logits[j][i-1] predicts token ids[j][i]
            logits_tensor = torch.tensor(logits[j][i - 1], dtype=torch.float32)
            token_id = ids[j][i]

            # Surprise: s(x_i) = -log p(x_i | context)
            log_probs = torch.log_softmax(beta * logits_tensor, dim=-1)
            s_i = -log_probs[token_id].item()

            # Conditional entropy: H_Q(X_i | context)
            probs = torch.exp(log_probs)
            h_i = -torch.sum(probs * log_probs).item()

            # Excess surprise: ε = s - H_Q
            eps_i = s_i - h_i

            eps_j.append(eps_i)
            s_j.append(s_i)
            h_j.append(h_i)

        epsilon_all.append(eps_j)
        surprise_all.append(s_j)
        h_cond_all.append(h_j)

    # Prompt trimming (same logic as spilled_energy)
    if prompt_length > 0:
        for j in range(len(epsilon_all)):
            for arr in [epsilon_all, surprise_all, h_cond_all]:
                if len(arr[j]) > prompt_length:
                    arr[j] = arr[j][prompt_length:]
                else:
                    arr[j] = arr[j][-1:]

    return epsilon_all, surprise_all, h_cond_all
```

### 2.6 `find_taxonomy_thresholds` — Automatic Threshold Selection

```python
def find_taxonomy_thresholds(
    delta_values: List[list],
    h_cond_values: List[list],
    is_correct: List[bool],
) -> Tuple[float, float]:
    """
    Find optimal (τ_δ, τ_h) thresholds for the diagnostic taxonomy
    using labeled validation data.

    Strategy: grid search over quantiles of |ΔE| and H_Q to maximize
    the F1 score for hallucination detection (high |ΔE| + low H_Q = hallucination).

    Args:
        delta_values: Nested list [n_samples][seq_len] — spilled energy.
        h_cond_values: Nested list [n_samples][seq_len] — conditional entropy.
        is_correct: List of booleans — ground truth correctness per sample.

    Returns:
        Tuple of (tau_delta, tau_h) — optimal thresholds.
    """
    from sklearn.metrics import f1_score as sklearn_f1

    # Pool per-sample: use mean |ΔE| and mean H_Q per sample
    sample_delta = []
    sample_h = []
    for j in range(len(delta_values)):
        vals = [abs(float(v.item() if hasattr(v, 'item') else v))
                for v in delta_values[j]]
        sample_delta.append(float(np.mean(vals)) if vals else 0.0)

        h_vals = [float(v) for v in h_cond_values[j]]
        sample_h.append(float(np.mean(h_vals)) if h_vals else 0.0)

    sample_delta = np.array(sample_delta)
    sample_h = np.array(sample_h)
    labels = np.array([not c for c in is_correct])  # True = hallucination

    if len(set(labels)) < 2:
        return float(np.median(sample_delta)), float(np.median(sample_h))

    best_f1 = -1.0
    best_tau = (float(np.median(sample_delta)), float(np.median(sample_h)))

    delta_quantiles = np.percentile(sample_delta, np.arange(10, 91, 10))
    h_quantiles = np.percentile(sample_h, np.arange(10, 91, 10))

    for td in delta_quantiles:
        for th in h_quantiles:
            # Predict hallucination: high |ΔE| AND low H_Q
            preds = (sample_delta > td) & (sample_h < th)
            f1 = sklearn_f1(labels, preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = (float(td), float(th))

    return best_tau
```

---

## Step 3: Update `__init__.py`

Expose the new public API:

```python
"""
Package for energy-based LLM detection.
"""

__all__ = [
    # Original energy functions
    "spilled_energy",
    "spilled_energy_torch",
    "spilled_energy_last_token",
    "compute_softmax_denominator",
    "compute_token_logit",
    # New: building blocks
    "compute_conditional_entropy",
    "compute_surprise",
    # New: MI-reframed features
    "mi_proxy",
    "excess_surprise",
    "excess_surprise_sequence",
    "calibrated_spilled_energy",
    "classify_tokens",
    "find_taxonomy_thresholds",
    # New: classification labels
    "HALLUCINATION",
    "UNCERTAIN",
    "CORRECT",
    "LUCKY_GUESS",
]
```

---

## Step 4: Benchmark Script `benchmark_mi.py`

Create `src/scripts/benchmark_mi.py` — extends the existing `benchmark_methods.py` to compare original spilled energy against the three new metrics.

### Design

The script will:
1. Reuse the same TriviaQA pipeline (generate → extract → locate tokens → compute)
2. Compute **all** metrics on the same token slices:
   - Original: delta, E, E_margin (existing)
   - New: excess_surprise, calibrated delta, MI proxy, H_Q
3. Evaluate each metric with the same 4 aggregation strategies (mean/max/min/sum)
4. Also run the 2D taxonomy classifier with auto-tuned thresholds
5. Report AUROC comparison table

```python
"""
Benchmark comparing original spilled energy vs MI-reframed metrics
on TriviaQA hallucination detection.
"""
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
import transformers
import datasets
from sklearn.metrics import roc_auc_score, precision_recall_curve

from spilled_energy.generation import generate_answer
from spilled_energy.extraction import extract_exact_answer
from spilled_energy.energy import spilled_energy
from spilled_energy.mi import (
    calibrated_spilled_energy,
    excess_surprise_sequence,
    classify_tokens,
    find_taxonomy_thresholds,
    HALLUCINATION,
)

# ... logging setup same as benchmark_methods.py ...


def compute_all_metrics(logits_tensor, generated_ids, token_start, token_end, alpha=1.0):
    """
    Compute original + MI-reframed metrics on a token slice.
    """
    sliced_logits = logits_tensor[:, token_start:token_end, :]
    sliced_ids = generated_ids[:, token_start:token_end]
    logits_list = sliced_logits.cpu().float().numpy().tolist()
    ids_list = sliced_ids.cpu().numpy().tolist()

    # Original metrics
    delta, E_margin, E = spilled_energy(logits=logits_list, ids=ids_list, beta=1.0)

    # New: excess surprise
    epsilon, surprise, h_cond_es = excess_surprise_sequence(
        logits=logits_list, ids=ids_list, beta=1.0
    )

    # New: calibrated spilled energy
    delta_cal, delta_raw, h_cond_cal, mi_proxy_vals = calibrated_spilled_energy(
        logits=logits_list, ids=ids_list, beta=1.0, alpha=alpha
    )

    def get_stats(vals):
        v = np.array([float(x.item() if hasattr(x, 'item') else x) for x in vals])
        if len(v) == 0:
            return {k: 0.0 for k in ["mean", "max", "min", "sum"]}
        return {
            "mean": float(np.mean(v)),
            "max": float(np.max(v)),
            "min": float(np.min(v)),
            "sum": float(np.sum(v)),
        }

    return {
        # Original
        "delta": get_stats(delta[0]),
        "E": get_stats(E[0]),
        "E_margin": get_stats(E_margin[0]),
        # MI-reframed
        "excess_surprise": get_stats(epsilon[0]),
        "delta_cal": get_stats(delta_cal[0]),
        "h_cond": get_stats(h_cond_cal[0]),
        "mi_proxy": get_stats(mi_proxy_vals[0]),
        # Raw lists for taxonomy
        "_delta_raw": delta,
        "_h_cond_raw": h_cond_cal,
    }


def main():
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    N_VAL = 25
    N_TEST = 50
    ALPHA = 1.0

    # ... model/dataset loading same as benchmark_methods.py ...
    # ... sample processing loop same as benchmark_methods.py,
    #     but calling compute_all_metrics() instead ...

    # === Analysis ===
    val_data = [r for r in results if r["split"] == "val"]
    test_data = [r for r in results if r["split"] == "test"]

    # --- Standard metric comparison ---
    metrics_to_eval = [
        "delta", "E", "E_margin",           # original
        "excess_surprise", "delta_cal",      # new
    ]
    # ... same AUROC evaluation loop as benchmark_methods.py ...

    # --- Taxonomy evaluation ---
    # Find thresholds on val
    val_delta = [r["metrics"]["_delta_raw"][0] for r in val_data]
    val_h = [r["metrics"]["_h_cond_raw"][0] for r in val_data]
    val_correct = [r["is_correct"] for r in val_data]
    tau_delta, tau_h = find_taxonomy_thresholds(val_delta, val_h, val_correct)
    print(f"\nTaxonomy thresholds: τ_δ={tau_delta:.2f}, τ_h={tau_h:.2f}")

    # Classify test samples
    test_preds = []
    for r in test_data:
        labels = classify_tokens(
            r["metrics"]["_delta_raw"],
            r["metrics"]["_h_cond_raw"],
            tau_delta, tau_h
        )
        # A sample is "hallucination" if any token is classified as such
        has_hallucination = any(l == HALLUCINATION for l in labels[0])
        test_preds.append(has_hallucination)

    test_labels = [not r["is_correct"] for r in test_data]
    from sklearn.metrics import f1_score, accuracy_score
    print(f"Taxonomy F1: {f1_score(test_labels, test_preds):.4f}")
    print(f"Taxonomy Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
```

---

## Step 5: Interactive Notebook `mi_diagnostic.ipynb`

Create `notebooks/mi_diagnostic.ipynb` with these sections:

### Cell structure:

1. **Setup** — imports, model loading (same pattern as `measure_exact_answer.ipynb`)
2. **Generate & Extract** — single TriviaQA sample
3. **Compute All Metrics** — show delta, delta_cal, epsilon, H_Q, MI_proxy side by side
4. **Token Visualization** — color-coded tokens using `plot_tokens` with calibrated delta
5. **2D Scatter Plot** — `|ΔE|` vs `H_Q` scatter with taxonomy quadrants drawn as colored regions
6. **Taxonomy Classification** — classify each token and display labeled output
7. **Comparison Histograms** — batch of samples showing calibrated vs uncalibrated delta distributions

Key visualization (cell 5) sketch:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter: each point is a token
ax.scatter(h_cond_vals, abs_delta_vals, c=colors_by_label, s=20, alpha=0.7)

# Draw quadrant boundaries
ax.axhline(y=tau_delta, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=tau_h, color='gray', linestyle='--', alpha=0.5)

# Label quadrants
ax.text(tau_h * 0.5, tau_delta * 1.5, 'HALLUCINATION', ha='center', color='red')
ax.text(tau_h * 1.5, tau_delta * 1.5, 'UNCERTAIN', ha='center', color='orange')
ax.text(tau_h * 0.5, tau_delta * 0.5, 'CORRECT', ha='center', color='green')
ax.text(tau_h * 1.5, tau_delta * 0.5, 'LUCKY GUESS', ha='center', color='gray')

ax.set_xlabel('Conditional Entropy H_Q')
ax.set_ylabel('|Spilled Energy ΔE|')
ax.set_title('Diagnostic Taxonomy')
```

---

## Step 6: Documentation

### 6.1 New file `documentation/mi_theory.md`

Contents:
- Conceptual explanation of the MI decomposition (EvoRate Proposition 1 in plain English)
- The three new metrics with formulas and intuition
- The false-positive problem and why calibration works
- The diagnostic taxonomy with interpretation guide
- Code examples for each metric
- Guidance on choosing `alpha` and thresholds

### 6.2 Update `documentation/api_reference.md`

Add a new section for the `spilled_energy.mi` module documenting all 6 public functions and the 4 classification constants.

---

## Implementation Order and Dependencies

```
Step 1: energy.py additions (compute_conditional_entropy, compute_surprise)
   ↓
Step 2: mi.py (all 6 functions — depends on Step 1)
   ↓
Step 3: __init__.py update (depends on Steps 1-2)
   ↓
Step 4: benchmark_mi.py (depends on Steps 1-3)
   ↓
Step 5: mi_diagnostic.ipynb (depends on Steps 1-3)
   ↓
Step 6: documentation (depends on Steps 1-5)
```

Steps 4 and 5 are independent of each other and can be done in parallel.

---

## Testing Strategy

No formal test framework exists in this repo, so follow the existing pattern:

1. **Smoke test via script:** Run `benchmark_mi.py` with small N_VAL=5, N_TEST=10 to verify the full pipeline works end-to-end.

2. **Numerical sanity checks to embed in the notebook:**
   - `H_Q` is always in `[0, log V]`
   - `MI_proxy` is always in `[0, log V]`
   - `|delta_cal| <= |delta_raw|` always (calibration only shrinks magnitude)
   - `surprise >= 0` always
   - `epsilon = surprise - H_Q` (verify identity)
   - When `H_Q ≈ 0`: `delta_cal ≈ delta_raw` (high-MI preservation)
   - When `H_Q ≈ log V`: `delta_cal ≈ delta_raw / (1 + α·log V)` (suppression)

3. **Regression check:** Original `spilled_energy()` output must be unchanged — the new code only *adds* functions, never modifies existing ones.

---

## Key Mathematical Identities to Verify in Code

These relationships from the paper should hold numerically (up to float precision):

| Identity | Paper Reference | How to Check |
|---|---|---|
| `s(x_i) = E_logit - E_margin_same_step` | Lemma 1, Eq. 18 | Compare `surprise` with `E[i] - E_margin[i-1]` (note index shift) |
| `ΔE ≈ ε + η` | Corollary 1, Eq. 29 | Compare `delta[i]` with `epsilon[i] + (E_margin[i-1] - E_margin[i])` |
| `s(x_i) = -log_softmax(logits)[token_id]` | Eq. 4 | Direct computation check |
| `H_Q = -Σ p·log(p)` | Eq. 16 | Compare with `scipy.stats.entropy` |

---

## Design Decisions and Rationale

**Q: Why a separate `mi.py` instead of adding everything to `energy.py`?**
A: The new features represent a distinct theoretical contribution (MI-reframing) that builds *on top of* the original energy computations. A separate module keeps the original API stable and makes the extension clearly identifiable. The `energy.py` additions (`compute_conditional_entropy`, `compute_surprise`) are genuine energy-level primitives that belong there.

**Q: Why list-based interfaces (matching `spilled_energy()`) instead of tensor-only?**
A: The entire existing codebase (all scripts, all notebooks) uses the list-based `spilled_energy()`. Matching this interface ensures the new features drop in seamlessly. Tensor-based variants can be added later following the `spilled_energy_torch` precedent.

**Q: Why default `alpha=1.0` for calibration?**
A: From Proposition 7(b): with α=1 and V=128,000 (LLaMA-3), the suppression factor at maximum entropy is ~12.8×, which is aggressive enough to suppress punctuation false positives without requiring tuning. The paper derives this as the natural scale.

**Q: Why use `mean |ΔE|` and `mean H_Q` per sample for threshold finding?**
A: Per-token thresholds would require token-level ground truth (which token is hallucinated), which standard QA benchmarks don't provide. Per-sample aggregation with mean matches the existing benchmark methodology.
