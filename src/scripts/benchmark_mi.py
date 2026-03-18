"""
Benchmark comparing original spilled energy vs MI-reframed metrics
on TriviaQA hallucination detection.

Extends benchmark_methods.py with: excess_surprise, calibrated delta,
MI proxy, conditional entropy, and the 2D diagnostic taxonomy.
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
import transformers
import datasets
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score

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

# Disable INFO logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()


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

    # Excess surprise
    epsilon, surprise, h_cond_es = excess_surprise_sequence(
        logits=logits_list, ids=ids_list, beta=1.0
    )

    # Calibrated spilled energy
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


def find_best_threshold(scores, labels):
    """
    Finds threshold that maximizes F1 score.
    labels: boolean, True = Hallucination (Positive class)
    """
    if len(set(labels)) < 2:
        return 0.0, 0.0

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = (
        thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    )
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


def main():
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    N_VAL = 25
    N_TEST = 50
    TOTAL_SAMPLES = N_VAL + N_TEST
    ALPHA = 1.0

    print(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16
        ).to("cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading TriviaQA (validation split)...")
    dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=True)

    results = []

    print(f"Processing {TOTAL_SAMPLES} samples...")

    iterator = iter(dataset)
    for i in tqdm(range(TOTAL_SAMPLES)):
        try:
            sample = next(iterator)
        except StopIteration:
            break

        question = sample["question"]
        ground_truth_aliases = sample["answer"]["aliases"]

        # 1. Generate
        prompt = f"Q: {question}\nA:"
        gen_output = generate_answer(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            do_sample=False,
            device="cuda",
        )
        generated_text = gen_output["text"]

        # 2. Extract
        exact_answer = extract_exact_answer(
            question=question,
            long_answer=generated_text,
            model=model,
            tokenizer=tokenizer,
            device="cuda",
        )
        cleaned_exact = exact_answer.strip("'\"").strip()

        # 3. Check Correctness
        is_correct = any(
            alias.lower() in cleaned_exact.lower() for alias in ground_truth_aliases
        )

        # 4. Locate Tokens
        start_idx = generated_text.find(cleaned_exact)
        token_start, token_end = None, None

        if start_idx != -1:
            end_idx = start_idx + len(cleaned_exact)
            enc = tokenizer(
                generated_text, return_offsets_mapping=True, add_special_tokens=False
            )
            for t_i, (s, e) in enumerate(enc.offset_mapping):
                if s >= start_idx and token_start is None:
                    token_start = t_i
                if s < end_idx:
                    token_end = t_i + 1

        # 5. Compute All Metrics
        logits = torch.stack(gen_output["scores"], dim=1)
        sequences = gen_output["sequences"]
        input_len = sequences.shape[1] - logits.shape[1]
        generated_ids = sequences[:, input_len:]

        if token_start is None or token_end is None:
            token_start, token_end = 0, generated_ids.shape[1]

        token_start = max(0, min(token_start, logits.shape[1] - 1))
        token_end = max(token_start + 1, min(token_end, logits.shape[1]))

        metrics = compute_all_metrics(
            logits, generated_ids, token_start, token_end, alpha=ALPHA
        )

        split_type = "val" if i < N_VAL else "test"
        results.append(
            {
                "split": split_type,
                "metrics": metrics,
                "is_correct": is_correct,
                "exact_answer": cleaned_exact,
                "ground_truth": ground_truth_aliases,
            }
        )

    # ===== Analysis =====
    val_data = [r for r in results if r["split"] == "val"]
    test_data = [r for r in results if r["split"] == "test"]

    print(
        f"\nCollected {len(val_data)} validation samples and {len(test_data)} test samples."
    )
    print(f"Val Accuracy: {np.mean([r['is_correct'] for r in val_data]):.2%}")
    print(f"Test Accuracy: {np.mean([r['is_correct'] for r in test_data]):.2%}")

    # --- Standard metric comparison ---
    print("\n--- Benchmarking Results (AUROC on Test Set) ---")
    print(f"{'Metric':<30} | {'Strategy':<10} | {'Val F1':<10} | {'Test AUROC':<10}")
    print("-" * 75)

    aggregated_results = []

    metrics_to_eval = [
        "delta", "E", "E_margin",            # original
        "excess_surprise", "delta_cal",       # MI-reframed
    ]

    for metric_name in metrics_to_eval:
        for strategy in ["mean", "max", "min", "sum"]:
            val_scores = np.array(
                [r["metrics"][metric_name][strategy] for r in val_data]
            )
            val_labels = np.array([not r["is_correct"] for r in val_data])

            test_scores = np.array(
                [r["metrics"][metric_name][strategy] for r in test_data]
            )
            test_labels = np.array([not r["is_correct"] for r in test_data])

            threshold, best_f1 = find_best_threshold(val_scores, val_labels)

            if len(set(test_labels)) > 1:
                auroc = roc_auc_score(test_labels, test_scores)
            else:
                auroc = 0.5

            print(
                f"{metric_name:<30} | {strategy:<10} | {best_f1:.4f}     | {auroc:.4f}"
            )
            aggregated_results.append((metric_name, strategy, auroc))

    best_metric = max(aggregated_results, key=lambda x: x[2])
    print("\nBest Performing Method:")
    print(f"{best_metric[0]} ({best_metric[1]}) - AUROC: {best_metric[2]:.4f}")

    # --- Taxonomy evaluation ---
    print("\n--- Diagnostic Taxonomy Evaluation ---")

    val_delta = [r["metrics"]["_delta_raw"][0] for r in val_data]
    val_h = [r["metrics"]["_h_cond_raw"][0] for r in val_data]
    val_correct = [r["is_correct"] for r in val_data]
    tau_delta, tau_h = find_taxonomy_thresholds(val_delta, val_h, val_correct)
    print(f"Thresholds: tau_delta={tau_delta:.4f}, tau_h={tau_h:.4f}")

    test_preds = []
    for r in test_data:
        labels = classify_tokens(
            r["metrics"]["_delta_raw"],
            r["metrics"]["_h_cond_raw"],
            tau_delta,
            tau_h,
        )
        has_hallucination = any(l == HALLUCINATION for l in labels[0])
        test_preds.append(has_hallucination)

    test_labels = [not r["is_correct"] for r in test_data]
    if len(set(test_labels)) > 1:
        print(f"Taxonomy F1:       {f1_score(test_labels, test_preds):.4f}")
        print(f"Taxonomy Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    else:
        print("Cannot evaluate: only one class in test set.")


if __name__ == "__main__":
    main()
