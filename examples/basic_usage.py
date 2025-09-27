"""
Basic usage example for the LLM Hallucination Mitigation Framework.

This example demonstrates how to use the framework to reduce hallucinations
in LLM responses through metacognitive self-correction.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hallucination_mitigator import HallucinationMitigator
from benchmark_suite import BenchmarkSuite


def example_openai_integration():
    """Example showing integration with OpenAI API."""
    try:
        import openai
    except ImportError:
        print("OpenAI package not installed. Run: pip install openai")
        return

    # Initialize the mitigator
    mitigator = HallucinationMitigator()

    # Set up OpenAI client (requires API key)
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def openai_response_function(prompt: str) -> str:
        """Function to get response from OpenAI."""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting response: {e}"

    # Example query that often leads to hallucinations
    query = "What are the exact binding affinities (Ki values in nM) of escitalopram for all serotonin receptor subtypes?"

    print("="*60)
    print("HALLUCINATION MITIGATION EXAMPLE")
    print("="*60)
    print(f"Query: {query}")
    print()

    # Apply the framework
    if os.getenv("OPENAI_API_KEY"):
        result = mitigator.apply_framework(query, openai_response_function)

        print("BASELINE RESPONSE:")
        print("-" * 40)
        print(result["baseline_response"])
        print()

        print("FRAMEWORK-GUIDED RESPONSE:")
        print("-" * 40)
        print(result["framework_response"])
        print()

        print("IMPROVEMENT METRICS:")
        print("-" * 40)
        improvement = result["improvement_metrics"]
        print(f"Hallucination Risk Reduction: {improvement['hallucination_risk_reduction']:.2f}")
        print(f"Evidence Coverage Improvement: {improvement['evidence_coverage_improvement']:.2f}")
        print(f"Uncertainty Acknowledgment Added: {improvement['uncertainty_acknowledgment_added']}")
    else:
        print("No OpenAI API key found. Set OPENAI_API_KEY environment variable to run this example.")


def example_anthropic_integration():
    """Example showing integration with Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        print("Anthropic package not installed. Run: pip install anthropic")
        return

    # Initialize the mitigator
    mitigator = HallucinationMitigator()

    # Set up Anthropic client (requires API key)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def claude_response_function(prompt: str) -> str:
        """Function to get response from Claude."""
        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error getting response: {e}"

    # Test with one of the benchmark queries
    query = "Please provide a list of 10 references related to social cognition in patients with schizophrenia. I'd like references that were published no earlier than 2022."

    print("="*60)
    print("CLAUDE HALLUCINATION MITIGATION EXAMPLE")
    print("="*60)

    if os.getenv("ANTHROPIC_API_KEY"):
        result = mitigator.apply_framework(query, claude_response_function)

        # Analyze the results
        baseline_analysis = result["baseline_analysis"]
        framework_analysis = result["framework_analysis"]

        print(f"BASELINE METRICS:")
        print(f"  Hallucination Risk Score: {baseline_analysis['metrics']['hallucination_risk_score']:.2f}")
        print(f"  Evidence Tag Coverage: {baseline_analysis['metrics']['evidence_tag_coverage']:.2f}")
        print(f"  Uncertainty Acknowledgment: {baseline_analysis['metrics']['uncertainty_acknowledgment']}")
        print()

        print(f"FRAMEWORK METRICS:")
        print(f"  Hallucination Risk Score: {framework_analysis['metrics']['hallucination_risk_score']:.2f}")
        print(f"  Evidence Tag Coverage: {framework_analysis['metrics']['evidence_tag_coverage']:.2f}")
        print(f"  Uncertainty Acknowledgment: {framework_analysis['metrics']['uncertainty_acknowledgment']}")
    else:
        print("No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable to run this example.")


def example_benchmark_suite():
    """Example of running the benchmark suite with a mock model."""
    print("="*60)
    print("BENCHMARK SUITE EXAMPLE")
    print("="*60)

    suite = BenchmarkSuite()

    # Mock model function for demonstration
    def mock_model_response(prompt: str) -> str:
        """Mock model that demonstrates typical baseline vs framework behavior."""
        if "epistemic humility" in prompt.lower():
            # Framework-guided response with uncertainty acknowledgment
            return """I need to approach this query with appropriate caution.

While I have general knowledge about this topic, I cannot provide specific recent citations with high confidence [N/E].

Instead, I can suggest:
1. Searching PubMed with relevant keywords
2. Checking recent issues of key journals in the field
3. Looking at review articles for current state of research

I acknowledge this limitation rather than providing potentially inaccurate specific references [N/E]."""
        else:
            # Baseline response with potential hallucinations
            return """Here are 10 relevant references:

1. Smith et al. (2023). "Social cognition deficits in schizophrenia" Journal of Psychiatric Research [DOI:10.1016/j.jpsychires.2023.01.001]
2. Johnson & Lee (2022). "Theory of mind in psychotic disorders" Schizophrenia Bulletin [DOI:10.1093/schbul/sb2022.156]
3. Brown et al. (2023). "Metacognitive training interventions" Psychological Medicine [DOI:10.1017/S0033291723000012]

[Note: These are example citations that may not be real references]"""

    # Run a single test case
    print("Running single test case...")
    result = suite.run_single_test("SZ_REFERENCES", mock_model_response)

    print(f"Test Query: {result['query']}")
    print()
    print("Baseline Response Metrics:")
    baseline_metrics = result["baseline_analysis"]["metrics"]
    print(f"  Risk Score: {baseline_metrics['hallucination_risk_score']:.2f}")
    print(f"  Uncertainty Acknowledgment: {baseline_metrics['uncertainty_acknowledgment']}")
    print()
    print("Framework Response Metrics:")
    framework_metrics = result["framework_analysis"]["metrics"]
    print(f"  Risk Score: {framework_metrics['hallucination_risk_score']:.2f}")
    print(f"  Uncertainty Acknowledgment: {framework_metrics['uncertainty_acknowledgment']}")


def example_custom_analysis():
    """Example of using the framework for custom response analysis."""
    print("="*60)
    print("CUSTOM RESPONSE ANALYSIS EXAMPLE")
    print("="*60)

    mitigator = HallucinationMitigator()

    # Example response with mixed evidence quality
    sample_response = """
    Recent studies have shown promising results for this treatment approach [DOI:10.1038/nm.2023.4567].

    The FDA has approved this medication [URL:https://www.fda.gov/drugs/news-events/fda-approves-new-treatment],
    and clinical trials demonstrated a 65% response rate [MEMORY:Phase III trial published in NEJM].

    However, long-term effects remain unclear [N/E], and more research is needed to establish optimal dosing protocols [N/E].

    I have high confidence in the approval status but acknowledge uncertainty about the long-term outcomes.
    """

    print("Analyzing sample response...")
    analysis = mitigator.analyze_response(sample_response)

    print("\nCLAIMS EXTRACTED:")
    for i, claim in enumerate(analysis["claims"], 1):
        print(f"{i}. {claim.text}")
        print(f"   Evidence: {claim.evidence_tag.value}")
        if claim.evidence_source:
            print(f"   Source: {claim.evidence_source}")
        print(f"   Confidence: {claim.confidence.value}")
        print()

    print("OVERALL METRICS:")
    metrics = analysis["metrics"]
    print(f"Total Claims: {metrics['total_claims']}")
    print(f"Hallucination Risk Score: {metrics['hallucination_risk_score']:.2f}")
    print(f"Evidence Tag Coverage: {metrics['evidence_tag_coverage']:.2f}")
    print(f"Uncertainty Acknowledgment: {metrics['uncertainty_acknowledgment']}")


if __name__ == "__main__":
    print("LLM Hallucination Mitigation Framework - Examples")
    print("=" * 60)

    # Run examples based on available API keys and packages
    example_custom_analysis()
    print("\n" + "="*60 + "\n")

    example_benchmark_suite()
    print("\n" + "="*60 + "\n")

    # Only run API examples if credentials are available
    if os.getenv("OPENAI_API_KEY"):
        example_openai_integration()
        print("\n" + "="*60 + "\n")

    if os.getenv("ANTHROPIC_API_KEY"):
        example_anthropic_integration()

    print("\nExample execution complete!")
    print("To run API examples, set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables.")