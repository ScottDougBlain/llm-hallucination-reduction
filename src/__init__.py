"""
LLM Hallucination Mitigation Framework

A metacognitive self-correction framework that applies cognitive behavioral therapy
principles to reduce LLM hallucinations through epistemic humility, evidence
gathering, verification, uncertainty calibration, and structured response generation.
"""

from .hallucination_mitigator import (
    HallucinationMitigator,
    Claim,
    EvidenceTag,
    ConfidenceLevel,
    VerificationQuestion
)
from .benchmark_suite import BenchmarkSuite, BenchmarkTest

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "HallucinationMitigator",
    "Claim",
    "EvidenceTag",
    "ConfidenceLevel",
    "VerificationQuestion",
    "BenchmarkSuite",
    "BenchmarkTest"
]