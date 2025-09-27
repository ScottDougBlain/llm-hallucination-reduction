"""
Tests for the hallucination mitigation framework.
"""

import pytest
from unittest.mock import Mock, patch
from src.hallucination_mitigator import (
    HallucinationMitigator,
    Claim,
    EvidenceTag,
    ConfidenceLevel,
    VerificationQuestion
)


class TestHallucinationMitigator:
    """Test suite for HallucinationMitigator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mitigator = HallucinationMitigator()

    def test_meta_prompt_structure(self):
        """Test that meta-prompt contains required components."""
        meta_prompt = self.mitigator.meta_prompt

        required_sections = [
            "epistemic humility",
            "Gather Evidence",
            "Verify",
            "Calibrate Uncertainty",
            "Produce Final Answer"
        ]

        for section in required_sections:
            assert section in meta_prompt, f"Meta-prompt missing section: {section}"

    def test_evidence_tag_extraction(self):
        """Test extraction of evidence tags from responses."""
        test_response = """
        This study found significant results [DOI:10.1234/example].
        The website confirms this [URL:https://example.com].
        I recall reading about this [MEMORY:specific research paper].
        However, I'm not certain about this claim [N/E].
        """

        claims = self.mitigator._extract_claims(test_response)

        # Should extract 4 claims with different evidence tags
        assert len(claims) >= 4

        # Check that different evidence types are present
        evidence_types = {claim.evidence_tag for claim in claims}
        expected_types = {EvidenceTag.DOI, EvidenceTag.URL, EvidenceTag.MEMORY, EvidenceTag.NO_EVIDENCE}
        assert expected_types.issubset(evidence_types)

    def test_confidence_inference(self):
        """Test confidence level inference from text."""
        # High confidence indicators
        high_conf_sentence = "This is definitely established research with proven results."
        confidence = self.mitigator._infer_confidence(high_conf_sentence, "")
        assert confidence == ConfidenceLevel.HIGH

        # Low confidence indicators
        low_conf_sentence = "This might possibly be true, but evidence is uncertain."
        confidence = self.mitigator._infer_confidence(low_conf_sentence, "")
        assert confidence == ConfidenceLevel.LOW

        # Medium confidence (default)
        medium_conf_sentence = "This research suggests interesting findings."
        confidence = self.mitigator._infer_confidence(medium_conf_sentence, "")
        assert confidence == ConfidenceLevel.MEDIUM

    def test_verification_questions_generation(self):
        """Test generation of verification questions."""
        test_response = "This is a test response with some claims."
        questions = self.mitigator._generate_verification_questions(test_response)

        assert len(questions) == 3

        focus_areas = {q.focus_area for q in questions}
        expected_areas = {"factual accuracy", "logical consistency", "completeness"}
        assert focus_areas == expected_areas

    def test_metrics_calculation(self):
        """Test calculation of hallucination metrics."""
        # Create test claims
        test_claims = [
            Claim("Test claim 1", EvidenceTag.DOI, "10.1234/test", ConfidenceLevel.HIGH),
            Claim("Test claim 2", EvidenceTag.NO_EVIDENCE, None, ConfidenceLevel.LOW),
            Claim("Test claim 3", EvidenceTag.URL, "https://test.com", ConfidenceLevel.MEDIUM),
        ]

        test_response = "This response acknowledges limitations and uncertainty."
        metrics = self.mitigator._calculate_metrics(test_claims, test_response)

        assert metrics["total_claims"] == 3
        assert metrics["hallucination_risk_score"] == pytest.approx(1/3, rel=1e-2)  # 1 high-risk claim out of 3
        assert metrics["evidence_tag_coverage"] == pytest.approx(2/3, rel=1e-2)  # 2 claims with evidence out of 3
        assert metrics["uncertainty_acknowledgment"] is True

    def test_empty_response_handling(self):
        """Test handling of empty or minimal responses."""
        empty_response = ""
        analysis = self.mitigator.analyze_response(empty_response)

        assert analysis["metrics"]["total_claims"] == 0
        assert analysis["metrics"]["hallucination_risk_score"] == 0.0

    def test_apply_framework_integration(self):
        """Test the complete framework application."""
        # Mock model response function
        def mock_model_response(prompt):
            if "epistemic humility" in prompt:
                return "I need to be careful here [N/E]. I don't have complete information about this topic."
            else:
                return "This is definitely true based on multiple studies [DOI:fake.reference]."

        query = "Test query about scientific facts"
        result = self.mitigator.apply_framework(query, mock_model_response)

        # Verify structure of result
        required_keys = [
            "query", "baseline_response", "framework_response",
            "baseline_analysis", "framework_analysis", "improvement_metrics"
        ]
        for key in required_keys:
            assert key in result

        # Verify improvement metrics structure
        improvement = result["improvement_metrics"]
        assert "hallucination_risk_reduction" in improvement
        assert "evidence_coverage_improvement" in improvement
        assert "uncertainty_acknowledgment_added" in improvement

    def test_benchmark_results_aggregation(self):
        """Test aggregation of benchmark results."""
        # Add some mock verification history
        mock_results = [
            {
                "improvement_metrics": {
                    "hallucination_risk_reduction": 0.5,
                    "evidence_coverage_improvement": 0.3,
                    "uncertainty_acknowledgment_added": True
                }
            },
            {
                "improvement_metrics": {
                    "hallucination_risk_reduction": 0.7,
                    "evidence_coverage_improvement": 0.2,
                    "uncertainty_acknowledgment_added": False
                }
            }
        ]

        self.mitigator.verification_history = mock_results
        benchmark_results = self.mitigator.get_benchmark_results()

        assert benchmark_results["total_evaluations"] == 2
        assert benchmark_results["average_hallucination_risk_reduction"] == 0.6
        assert benchmark_results["average_evidence_coverage_improvement"] == 0.25
        assert benchmark_results["uncertainty_acknowledgment_improvement_rate"] == 0.5


class TestClaimExtraction:
    """Specific tests for claim extraction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mitigator = HallucinationMitigator()

    def test_complex_citation_extraction(self):
        """Test extraction of academic citations with DOI tags."""
        response = """
        Recent research has shown significant improvements [DOI:10.1038/nature.2023.12345].
        Another study confirms this finding [DOI:10.1126/science.2023.67890].
        However, some claims lack verification [N/E].
        """

        claims = self.mitigator._extract_claims(response)
        doi_claims = [c for c in claims if c.evidence_tag == EvidenceTag.DOI]

        assert len(doi_claims) == 2
        assert "10.1038/nature.2023.12345" in [c.evidence_source for c in doi_claims]
        assert "10.1126/science.2023.67890" in [c.evidence_source for c in doi_claims]

    def test_url_reference_extraction(self):
        """Test extraction of URL references."""
        response = """
        According to the official documentation [URL:https://docs.example.com/api].
        See also this resource [URL:https://github.com/example/repo].
        """

        claims = self.mitigator._extract_claims(response)
        url_claims = [c for c in claims if c.evidence_tag == EvidenceTag.URL]

        assert len(url_claims) == 2
        urls = [c.evidence_source for c in url_claims]
        assert "https://docs.example.com/api" in urls
        assert "https://github.com/example/repo" in urls

    def test_mixed_evidence_types(self):
        """Test handling of responses with mixed evidence types."""
        response = """
        This is documented in the literature [DOI:10.1234/test].
        The website also mentions this [URL:https://test.com].
        I remember reading about this [MEMORY:specific paper title].
        Some aspects remain unclear [N/E].
        """

        claims = self.mitigator._extract_claims(response)
        evidence_types = {claim.evidence_tag for claim in claims}

        expected_types = {
            EvidenceTag.DOI,
            EvidenceTag.URL,
            EvidenceTag.MEMORY,
            EvidenceTag.NO_EVIDENCE
        }
        assert evidence_types == expected_types


if __name__ == "__main__":
    pytest.main([__file__])