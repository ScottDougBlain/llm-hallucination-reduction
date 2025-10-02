"""
LLM Hallucination Mitigation Framework

A metacognitive self-correction framework that applies cognitive behavioral therapy
principles to reduce LLM hallucinations through epistemic humility, evidence
gathering, verification, uncertainty calibration, and structured response generation.

Based on research showing 71% reduction in hallucination rates (from 7/7 to 2/7)
when applied to Claude 3.7 test queries.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re


class ConfidenceLevel(Enum):
    """Confidence levels for claims with specific thresholds."""
    HIGH = "High (≥90%)"
    MEDIUM = "Medium (60-89%)"
    LOW = "Low (<60%)"


class EvidenceTag(Enum):
    """Evidence tags for source verification."""
    URL = "URL"
    DOI = "DOI"
    PMID = "PMID"
    MEMORY = "MEMORY"
    NO_EVIDENCE = "N/E"


@dataclass
class Claim:
    """Represents a factual claim with evidence and confidence."""
    text: str
    evidence_tag: EvidenceTag
    evidence_source: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    verification_notes: Optional[str] = None


@dataclass
class VerificationQuestion:
    """Self-verification question for response validation."""
    question: str
    focus_area: str  # "factual accuracy", "logical consistency", or "completeness"
    answer: bool
    rationale: str


class HallucinationMitigator:
    """
    Implements metacognitive self-correction framework for reducing LLM hallucinations.

    This framework applies a five-step process:
    1. Epistemic humility stance
    2. Evidence gathering with source tagging
    3. Self-verification through targeted questioning
    4. Uncertainty calibration with confidence levels
    5. Final answer production with visible uncertainty
    """

    def __init__(self):
        self.meta_prompt = self._load_meta_prompt()
        self.verification_history: List[Dict] = []

    def _load_meta_prompt(self) -> str:
        """Load the metacognitive self-correction meta-prompt."""
        return """SYSTEM—Metacognitive Self-Correction
Before replying, complete the following internal steps:

1. Take a stance of epistemic humility
   • Prioritize factual accuracy and honesty over generating a plausible-sounding response
   • Consider alternative explanations and interpretations of available evidence
   • Recognize domains where your knowledge may be incomplete or outdated

2. Gather Evidence
   • For every factual claim, retrieve supporting information
   • Append one of the following tags after each claim:
     o [URL:] - For web-verifiable information
     o [DOI:] or [PMID:] - For academic sources
     o [MEMORY:] - For a specific, verifiable memory with concrete details
     o [N/E] - "No Evidence" when verification isn't possible
   • When evidence conflicts, note this explicitly

3. Verify
   • Ask yourself three specific yes/no questions that could invalidate your response
   • Focus on: (1) factual accuracy, (2) logical consistency, and (3) completeness
   • If any answer contradicts the draft, revise the problematic section

4. Calibrate Uncertainty
   • Assign confidence levels to each claim:
     o High (≥90%) - Multiple reliable sources or fundamental knowledge
     o Medium (60-89%) - Single reliable source or strong reasoning
     o Low (<60%) - Limited, conflicting, or unclear evidence
   • Be explicit about the source of uncertainty when present

5. Produce Final Answer
   • Keep tags and confidence labels visible to the user
   • Omit any statement that carries an [N/E] or low confidence tag unless explicitly asked for speculation
   • Provide alternative recommendations when specific factual information couldn't be verified
   • Summarize key uncertainties at the conclusion when relevant

Prompt:
[Insert specific query]"""

    def analyze_response(self, response: str) -> Dict[str, Union[List[Claim], List[VerificationQuestion], Dict]]:
        """
        Analyze a response for claims, evidence tags, and confidence indicators.

        Args:
            response: The LLM response to analyze

        Returns:
            Analysis dictionary containing claims, verification questions, and metrics
        """
        claims = self._extract_claims(response)
        verification_questions = self._generate_verification_questions(response)
        metrics = self._calculate_metrics(claims, response)

        return {
            "claims": claims,
            "verification_questions": verification_questions,
            "metrics": metrics,
            "raw_response": response
        }

    def _extract_claims(self, response: str) -> List[Claim]:
        """Extract factual claims with evidence tags from response."""
        claims = []

        # Pattern to match evidence tags
        tag_patterns = {
            EvidenceTag.URL: r'\[URL:([^\]]*)\]',
            EvidenceTag.DOI: r'\[DOI:([^\]]*)\]',
            EvidenceTag.PMID: r'\[PMID:([^\]]*)\]',
            EvidenceTag.MEMORY: r'\[MEMORY:([^\]]*)\]',
            EvidenceTag.NO_EVIDENCE: r'\[N/E\]'
        }

        # Split response into sentences for analysis
        sentences = re.split(r'[.!?]+', response)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check for evidence tags
            for tag_type, pattern in tag_patterns.items():
                matches = re.findall(pattern, sentence)
                if matches:
                    # Extract claim text (sentence without the tag)
                    claim_text = re.sub(pattern, '', sentence).strip()

                    # Determine confidence level from surrounding context
                    confidence = self._infer_confidence(sentence, response)

                    # Create claim object
                    evidence_source = matches[0] if matches and matches[0] else None
                    claims.append(Claim(
                        text=claim_text,
                        evidence_tag=tag_type,
                        evidence_source=evidence_source,
                        confidence=confidence
                    ))

        return claims

    def _infer_confidence(self, sentence: str, full_response: str) -> ConfidenceLevel:
        """Infer confidence level from textual indicators."""
        sentence_lower = sentence.lower()

        # High confidence indicators
        high_confidence_terms = ['definitely', 'certainly', 'clearly', 'established', 'proven']
        if any(term in sentence_lower for term in high_confidence_terms):
            return ConfidenceLevel.HIGH

        # Low confidence indicators
        low_confidence_terms = ['might', 'possibly', 'uncertain', 'unclear', 'limited evidence']
        if any(term in sentence_lower for term in low_confidence_terms):
            return ConfidenceLevel.LOW

        # Check for explicit confidence labels in the full response
        if 'High (' in full_response and sentence in full_response:
            return ConfidenceLevel.HIGH
        elif 'Low (' in full_response and sentence in full_response:
            return ConfidenceLevel.LOW

        return ConfidenceLevel.MEDIUM

    def _generate_verification_questions(self, response: str) -> List[VerificationQuestion]:
        """Generate verification questions for the three focus areas.

        Note: In production, these would be evaluated by the LLM or external verification.
        Current implementation provides heuristic-based assessment.
        """
        # Simple heuristic checks for demonstration
        has_evidence_tags = bool(re.search(r'\[(?:URL|DOI|PMID|MEMORY|N/E):[^\]]*\]', response))
        has_confidence_markers = any(marker in response.lower()
                                    for marker in ['high confidence', 'medium confidence',
                                                 'low confidence', 'uncertain'])

        questions = [
            VerificationQuestion(
                question="Are all factual claims in this response accurate and verifiable?",
                focus_area="factual accuracy",
                answer=has_evidence_tags,  # Heuristic: presence of evidence tags
                rationale="Evidence tags present" if has_evidence_tags else "No evidence tags found"
            ),
            VerificationQuestion(
                question="Is the logical flow of the argument consistent throughout?",
                focus_area="logical consistency",
                answer=len(response) > 100,  # Heuristic: sufficient length for structured response
                rationale="Response appears structured" if len(response) > 100 else "Response too brief"
            ),
            VerificationQuestion(
                question="Does this response address all aspects of the original query?",
                focus_area="completeness",
                answer=has_confidence_markers,  # Heuristic: includes uncertainty calibration
                rationale="Includes confidence assessment" if has_confidence_markers else "Missing confidence markers"
            )
        ]
        return questions

    def _calculate_metrics(self, claims: List[Claim], response: str) -> Dict:
        """Calculate key metrics for hallucination assessment."""
        total_claims = len(claims)

        if total_claims == 0:
            return {
                "total_claims": 0,
                "hallucination_risk_score": 0.0,
                "evidence_tag_coverage": 0.0,
                "uncertainty_acknowledgment": "limitations disclaimer" in response.lower()
            }

        # Count claims by evidence type
        evidence_counts = {}
        for tag in EvidenceTag:
            evidence_counts[tag.value] = sum(1 for claim in claims if claim.evidence_tag == tag)

        # Calculate hallucination risk (higher for N/E tags and low confidence)
        high_risk_claims = sum(1 for claim in claims
                              if claim.evidence_tag == EvidenceTag.NO_EVIDENCE
                              or claim.confidence == ConfidenceLevel.LOW)

        hallucination_risk_score = high_risk_claims / total_claims if total_claims > 0 else 0

        # Evidence tag coverage (percentage of claims with evidence)
        evidence_tag_coverage = sum(1 for claim in claims
                                  if claim.evidence_tag != EvidenceTag.NO_EVIDENCE) / total_claims

        return {
            "total_claims": total_claims,
            "evidence_counts": evidence_counts,
            "hallucination_risk_score": hallucination_risk_score,
            "evidence_tag_coverage": evidence_tag_coverage,
            "uncertainty_acknowledgment": any(phrase in response.lower()
                                            for phrase in ["i don't have", "i'm not certain", "limitations", "uncertain"])
        }

    def apply_framework(self, query: str, model_response_fn) -> Dict:
        """
        Apply the complete metacognitive framework to a query.

        Args:
            query: The original query
            model_response_fn: Function that takes a prompt and returns model response

        Returns:
            Dictionary with original response, framework-guided response, and analysis
        """
        # Get baseline response
        baseline_response = model_response_fn(query)

        # Apply meta-prompt
        framework_prompt = self.meta_prompt.replace("[Insert specific query]", query)
        framework_response = model_response_fn(framework_prompt)

        # Analyze both responses
        baseline_analysis = self.analyze_response(baseline_response)
        framework_analysis = self.analyze_response(framework_response)

        # Store in history
        result = {
            "query": query,
            "baseline_response": baseline_response,
            "framework_response": framework_response,
            "baseline_analysis": baseline_analysis,
            "framework_analysis": framework_analysis,
            "improvement_metrics": self._calculate_improvement(baseline_analysis, framework_analysis)
        }

        self.verification_history.append(result)
        return result

    def _calculate_improvement(self, baseline: Dict, framework: Dict) -> Dict:
        """Calculate improvement metrics between baseline and framework responses."""
        baseline_metrics = baseline["metrics"]
        framework_metrics = framework["metrics"]

        return {
            "hallucination_risk_reduction": baseline_metrics["hallucination_risk_score"] - framework_metrics["hallucination_risk_score"],
            "evidence_coverage_improvement": framework_metrics["evidence_tag_coverage"] - baseline_metrics["evidence_tag_coverage"],
            "uncertainty_acknowledgment_added": framework_metrics["uncertainty_acknowledgment"] and not baseline_metrics["uncertainty_acknowledgment"]
        }

    def get_benchmark_results(self) -> Dict:
        """Get summary statistics across all verified responses."""
        if not self.verification_history:
            return {"error": "No verification history available"}

        improvements = [result["improvement_metrics"] for result in self.verification_history]

        avg_risk_reduction = sum(imp["hallucination_risk_reduction"] for imp in improvements) / len(improvements)
        avg_evidence_improvement = sum(imp["evidence_coverage_improvement"] for imp in improvements) / len(improvements)
        uncertainty_acknowledgment_rate = sum(1 for imp in improvements if imp["uncertainty_acknowledgment_added"]) / len(improvements)

        return {
            "total_evaluations": len(self.verification_history),
            "average_hallucination_risk_reduction": avg_risk_reduction,
            "average_evidence_coverage_improvement": avg_evidence_improvement,
            "uncertainty_acknowledgment_improvement_rate": uncertainty_acknowledgment_rate
        }