"""
Tests for the benchmark suite functionality.
"""

import pytest
from unittest.mock import Mock, patch
import json
import tempfile
from pathlib import Path

from src.benchmark_suite import BenchmarkSuite, BenchmarkTest


class TestBenchmarkSuite:
    """Test suite for BenchmarkSuite class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.suite = BenchmarkSuite()

    def test_test_cases_loading(self):
        """Test that all expected test cases are loaded."""
        assert len(self.suite.test_cases) == 7

        # Verify all required test case IDs are present
        expected_ids = [
            "SZ_REFERENCES", "DRUG_BINDING", "EU_AI_ACT",
            "AI_LEADERBOARD", "TRUTHFULNESS_METRICS", "AI_QUOTES", "PUBLIC_PERCEPTION"
        ]

        actual_ids = [test.id for test in self.suite.test_cases]
        assert set(actual_ids) == set(expected_ids)

    def test_test_case_structure(self):
        """Test that test cases have proper structure."""
        for test_case in self.suite.test_cases:
            assert isinstance(test_case, BenchmarkTest)
            assert test_case.id
            assert test_case.query
            assert test_case.domain
            assert test_case.difficulty in ["easy", "medium", "hard"]
            assert test_case.expected_hallucination_risk in ["low", "medium", "high"]

    def test_get_test_case(self):
        """Test retrieval of specific test cases."""
        # Test valid test case
        test_case = self.suite.get_test_case("SZ_REFERENCES")
        assert test_case is not None
        assert test_case.id == "SZ_REFERENCES"
        assert "schizophrenia" in test_case.query.lower()

        # Test invalid test case
        assert self.suite.get_test_case("NONEXISTENT") is None

    def test_single_test_execution(self):
        """Test running a single test case."""
        def mock_model_response(prompt):
            if "epistemic humility" in prompt:
                return "I don't have reliable information about this [N/E]."
            else:
                return "Here are the references: [DOI:fake.reference]"

        result = self.suite.run_single_test("PUBLIC_PERCEPTION", mock_model_response)

        # Verify result structure
        assert "query" in result
        assert "baseline_response" in result
        assert "framework_response" in result
        assert "test_case" in result
        assert result["test_case"].id == "PUBLIC_PERCEPTION"

    def test_benchmark_statistics_calculation(self):
        """Test calculation of benchmark statistics."""
        # Create mock results
        mock_results = [
            {
                "baseline_analysis": {
                    "metrics": {
                        "hallucination_risk_score": 0.8,
                        "uncertainty_acknowledgment": False
                    }
                },
                "framework_analysis": {
                    "metrics": {
                        "hallucination_risk_score": 0.2,
                        "uncertainty_acknowledgment": True
                    }
                },
                "improvement_metrics": {
                    "hallucination_risk_reduction": 0.6,
                    "evidence_coverage_improvement": 0.3
                }
            },
            {
                "baseline_analysis": {
                    "metrics": {
                        "hallucination_risk_score": 0.9,
                        "uncertainty_acknowledgment": False
                    }
                },
                "framework_analysis": {
                    "metrics": {
                        "hallucination_risk_score": 0.1,
                        "uncertainty_acknowledgment": True
                    }
                },
                "improvement_metrics": {
                    "hallucination_risk_reduction": 0.8,
                    "evidence_coverage_improvement": 0.4
                }
            }
        ]

        stats = self.suite._calculate_benchmark_stats(mock_results)

        # Test calculation accuracy
        assert stats["baseline_hallucination_rate"] == 1.0  # Both above 0.5 threshold
        assert stats["framework_hallucination_rate"] == 0.0  # Both below 0.5 threshold
        assert stats["hallucination_reduction"] == 1.0  # 100% reduction
        assert stats["framework_uncertainty_rate"] == 1.0  # Both have uncertainty acknowledgment
        assert stats["average_risk_reduction"] == 0.7  # (0.6 + 0.8) / 2

    def test_benchmark_output_saving(self):
        """Test saving benchmark results to file."""
        mock_results = {
            "statistics": {"test": "data"},
            "detailed_results": [
                {
                    "test_case": BenchmarkTest(
                        id="TEST", query="test", domain="test",
                        difficulty="easy", expected_hallucination_risk="low"
                    ),
                    "other_data": "value"
                }
            ],
            "test_metadata": {"version": "1.0"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            self.suite._save_results(mock_results, temp_path)

            # Verify file was created and contains data
            assert Path(temp_path).exists()

            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            assert "statistics" in loaded_data
            assert "detailed_results" in loaded_data
            assert loaded_data["test_metadata"]["version"] == "1.0"

        finally:
            Path(temp_path).unlink()

    def test_full_benchmark_integration(self):
        """Test running the complete benchmark suite."""
        call_count = 0

        def mock_model_response(prompt):
            nonlocal call_count
            call_count += 1

            # Simulate different responses for baseline vs framework
            if "epistemic humility" in prompt:
                return f"Framework response {call_count} [N/E] with uncertainty acknowledgment."
            else:
                return f"Baseline response {call_count} [DOI:fake.reference.{call_count}]"

        # Run benchmark without saving results
        results = self.suite.run_benchmark(mock_model_response)

        # Verify structure
        assert "statistics" in results
        assert "detailed_results" in results
        assert "test_metadata" in results

        # Verify all test cases were run
        assert len(results["detailed_results"]) == 7
        assert results["statistics"]["test_cases_run"] == 7

        # Verify model was called for both baseline and framework for each test
        assert call_count == 14  # 7 tests × 2 calls each

    def test_domain_coverage(self):
        """Test that benchmark covers diverse domains."""
        domains = {test.domain for test in self.suite.test_cases}

        expected_domains = {
            "Academic Literature", "Pharmacology", "Legal/Regulatory", "AI Safety"
        }

        assert expected_domains.issubset(domains)

    def test_difficulty_distribution(self):
        """Test that benchmark includes various difficulty levels."""
        difficulties = [test.difficulty for test in self.suite.test_cases]

        # Should have mix of difficulties
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

        # Hard cases should be majority (challenging for hallucination detection)
        hard_count = difficulties.count("hard")
        assert hard_count >= 3


class TestBenchmarkTestDataclass:
    """Test the BenchmarkTest dataclass."""

    def test_benchmark_test_creation(self):
        """Test creation of BenchmarkTest objects."""
        test = BenchmarkTest(
            id="TEST_ID",
            query="Test query?",
            domain="Test Domain",
            difficulty="medium",
            expected_hallucination_risk="high",
            description="Test description"
        )

        assert test.id == "TEST_ID"
        assert test.query == "Test query?"
        assert test.domain == "Test Domain"
        assert test.difficulty == "medium"
        assert test.expected_hallucination_risk == "high"
        assert test.description == "Test description"

    def test_benchmark_test_optional_fields(self):
        """Test BenchmarkTest with optional fields."""
        test = BenchmarkTest(
            id="TEST",
            query="Query",
            domain="Domain",
            difficulty="easy",
            expected_hallucination_risk="low"
        )

        # Optional fields should have defaults
        assert test.baseline_url is None
        assert test.framework_url is None
        assert test.description == ""


if __name__ == "__main__":
    pytest.main([__file__])