"""
Package for energy-based LLM detection.
"""

from spilled_energy.energy import (
    compute_conditional_entropy,
    compute_softmax_denominator,
    compute_surprise,
    compute_token_logit,
    spilled_energy,
    spilled_energy_last_token,
    spilled_energy_torch,
)
from spilled_energy.mi import (
    CORRECT,
    HALLUCINATION,
    LUCKY_GUESS,
    UNCERTAIN,
    calibrated_spilled_energy,
    classify_tokens,
    excess_surprise,
    excess_surprise_sequence,
    find_taxonomy_thresholds,
    mi_proxy,
)

__all__ = [
    # Original energy functions
    "spilled_energy",
    "spilled_energy_torch",
    "spilled_energy_last_token",
    "compute_softmax_denominator",
    "compute_token_logit",
    # Building blocks
    "compute_conditional_entropy",
    "compute_surprise",
    # MI-reframed features
    "mi_proxy",
    "excess_surprise",
    "excess_surprise_sequence",
    "calibrated_spilled_energy",
    "classify_tokens",
    "find_taxonomy_thresholds",
    # Classification labels
    "HALLUCINATION",
    "UNCERTAIN",
    "CORRECT",
    "LUCKY_GUESS",
]
