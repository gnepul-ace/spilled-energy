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

# Token classification labels
HALLUCINATION = "hallucination"
UNCERTAIN = "uncertain"
CORRECT = "correct"
LUCKY_GUESS = "lucky_guess"


def mi_proxy(
    logits: torch.Tensor, beta: float = 1.0, vocab_size: Optional[int] = None
) -> torch.Tensor:
    """
    Compute MI proxy: a training-free lower bound on mutual information
    between the next token and the context.

    MI_proxy(i) = max(0, log(V) - H_Q(X_{i+1} | x_{i:1}))

    Args:
        logits: Tensor of shape (..., vocab_size).
        beta: Inverse temperature (default 1.0).
        vocab_size: Override vocabulary size V. If None, inferred from logits.

    Returns:
        Tensor of shape (...) — MI proxy values in nats. Range [0, log V].
    """
    if vocab_size is None:
        vocab_size = logits.shape[-1]
    log_v = torch.tensor(float(np.log(vocab_size)), device=logits.device)
    h_cond = compute_conditional_entropy(logits, beta=beta)
    return torch.clamp(log_v - h_cond, min=0.0)


def excess_surprise(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute excess surprise: how much more surprising the chosen token is
    than the model expected.

    ε(x_i) = s(x_i) - H_Q(X_i | x_{i-1:1})

    From the paper: E[ε] ≈ KL(P||Q), so excess surprise directly
    measures model error at each token position.

    Args:
        logits: Tensor of shape (..., vocab_size).
        token_ids: Tensor of shape (...) — the chosen token IDs.
        beta: Inverse temperature (default 1.0).

    Returns:
        Tuple of (epsilon, surprise, h_cond).
    """
    s = compute_surprise(logits, token_ids, beta=beta)
    h_cond = compute_conditional_entropy(logits, beta=beta)
    epsilon = s - h_cond
    return epsilon, s, h_cond


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

    Args:
        logits: Nested list [batch][seq_len][vocab_size].
        ids: Nested list [batch][seq_len].
        beta: Inverse temperature (default 1.0).
        prompt_length: Number of prompt tokens to exclude.

    Returns:
        Tuple of (epsilon, surprise, h_cond) — nested lists [batch][seq_len].
    """
    epsilon_all = []
    surprise_all = []
    h_cond_all = []

    for j in range(len(logits)):
        eps_j = [0.0]
        s_j = [0.0]
        h_j = [0.0]

        for i in range(1, len(logits[j])):
            logits_tensor = torch.tensor(logits[j][i - 1], dtype=torch.float32)
            token_id = ids[j][i]

            log_probs = torch.log_softmax(beta * logits_tensor, dim=-1)
            s_i = -log_probs[token_id].item()

            probs = torch.exp(log_probs)
            h_i = -torch.sum(probs * log_probs).item()

            eps_j.append(s_i - h_i)
            s_j.append(s_i)
            h_j.append(h_i)

        epsilon_all.append(eps_j)
        surprise_all.append(s_j)
        h_cond_all.append(h_j)

    if prompt_length > 0:
        for j in range(len(epsilon_all)):
            for arr in [epsilon_all, surprise_all, h_cond_all]:
                if len(arr[j]) > prompt_length:
                    arr[j] = arr[j][prompt_length:]
                else:
                    arr[j] = arr[j][-1:]

    return epsilon_all, surprise_all, h_cond_all


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

    At high-MI positions (H_Q ≈ 0), the raw signal is preserved.
    At low-MI positions (H_Q ≈ log V), the signal is suppressed by
    factor ~(1 + α·log V), addressing false positives on punctuation
    and sentence-initial tokens.

    Args:
        logits: Nested list [batch][seq_len][vocab_size].
        ids: Nested list [batch][seq_len].
        beta: Inverse temperature (default 1.0).
        alpha: Calibration strength (default 1.0).
        prompt_length: Number of prompt tokens to exclude.

    Returns:
        Tuple of (delta_cal, delta_raw, h_cond, mi_proxy_vals).
    """
    delta_raw, E_margin, E = spilled_energy(
        logits=logits, ids=ids, beta=beta, prompt_length=prompt_length
    )

    vocab_size = len(logits[0][0]) if logits and logits[0] else 1
    log_v = float(np.log(vocab_size))

    h_cond_all = []
    mi_proxy_all = []

    for j in range(len(logits)):
        h_cond_j = []
        mi_proxy_j = []

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


def classify_tokens(
    delta: list,
    h_cond: list,
    tau_delta: float,
    tau_h: float,
) -> List[List[str]]:
    """
    Classify each token into the diagnostic taxonomy.

    |ΔE| > τ_δ  AND  H_Q < τ_h   →  HALLUCINATION
    |ΔE| > τ_δ  AND  H_Q >= τ_h  →  UNCERTAIN
    |ΔE| <= τ_δ AND  H_Q < τ_h   →  CORRECT
    |ΔE| <= τ_δ AND  H_Q >= τ_h  →  LUCKY GUESS

    Args:
        delta: Nested list [batch][seq_len] — spilled energy values.
        h_cond: Nested list [batch][seq_len] — conditional entropy values.
        tau_delta: Threshold on |ΔE|.
        tau_h: Threshold on H_Q.

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
            elif d > tau_delta:
                labels_j.append(UNCERTAIN)
            elif h < tau_h:
                labels_j.append(CORRECT)
            else:
                labels_j.append(LUCKY_GUESS)

        labels.append(labels_j)
    return labels


def find_taxonomy_thresholds(
    delta_values: List[list],
    h_cond_values: List[list],
    is_correct: List[bool],
) -> Tuple[float, float]:
    """
    Find optimal (τ_δ, τ_h) thresholds for the diagnostic taxonomy
    using labeled validation data via grid search over quantiles.

    Args:
        delta_values: Nested list [n_samples][seq_len] — spilled energy.
        h_cond_values: Nested list [n_samples][seq_len] — conditional entropy.
        is_correct: List of booleans — ground truth correctness per sample.

    Returns:
        Tuple of (tau_delta, tau_h) — optimal thresholds.
    """
    from sklearn.metrics import f1_score as sklearn_f1

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
            preds = (sample_delta > td) & (sample_h < th)
            f1 = sklearn_f1(labels, preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = (float(td), float(th))

    return best_tau
