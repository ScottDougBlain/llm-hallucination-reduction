# LLM Hallucination Mitigation Framework

A structured prompting framework that implements metacognitive self-correction strategies to reduce LLM hallucinations through evidence tagging, verification questions, and uncertainty calibration.

## Preliminary Results

**Initial Testing on 7 Challenging Queries**: In preliminary testing with Claude 3.7 on a small benchmark suite of 7 queries specifically designed to provoke hallucinations, our structured prompting approach showed promising results, with hallucination rates dropping from 7/7 to 2/7 and uncertainty acknowledgment improving from 0/7 to 7/7.

| Condition   | Hallucination Rate | Uncertainty Acknowledgment | Mean Response Time |
|-------------|-------------------|----------------------------|-------------------|
| Baseline    | 7/7 (100%)        | 0/7 (0%)                  | 9.6s              |
| Framework   | 2/7 (29%)         | 7/7 (100%)                | 10.4s             |

**Important Note**: These are preliminary results from a small proof-of-concept evaluation. Rigorous testing with larger sample sizes, diverse query types, and multiple models would be needed to validate effectiveness claims.

## Conceptual Inspiration

This framework draws loose inspiration from psychological concepts including metacognition, reality-testing, and uncertainty tolerance. While we reference ideas from CBT and cognitive psychology, this is fundamentally a structured prompting technique rather than a clinical intervention.

### Core Psychological Principles

- **Epistemic Humility**: Counteracts overconfidence bias that leads to fabricated "knowledge"
- **Evidence-Based Reasoning**: Mirrors CBT techniques for reality testing
- **Metacognitive Awareness**: Implements self-monitoring processes found effective in clinical populations
- **Uncertainty Tolerance**: Reduces the drive to generate plausible-sounding but false information

## Framework Architecture

### Five-Step Metacognitive Process

1. **Epistemic Humility Stance**
   - Prioritize factual accuracy over plausible-sounding responses
   - Consider alternative explanations and interpretations
   - Recognize knowledge limitations and outdated information

2. **Evidence Gathering**
   - Tag every factual claim with evidence sources:
     - `[URL:]` - Web-verifiable information
     - `[DOI:]` or `[PMID:]` - Academic sources
     - `[MEMORY:]` - Specific verifiable memories
     - `[N/E]` - No evidence available
   - Explicitly note conflicting evidence

3. **Verification**
   - Three targeted yes/no questions:
     - Factual accuracy check
     - Logical consistency review
     - Completeness assessment
   - Revise problematic sections if any question fails

4. **Uncertainty Calibration**
   - Assign confidence levels:
     - **High (≥90%)**: Multiple reliable sources or fundamental knowledge
     - **Medium (60-89%)**: Single reliable source or strong reasoning
     - **Low (<60%)**: Limited, conflicting, or unclear evidence

5. **Final Answer Production**
   - Keep evidence tags and confidence labels visible
   - Omit low-confidence claims unless speculation is requested
   - Provide alternatives when verification fails
   - Summarize key uncertainties

## Quick Start

### Installation

```bash
git clone https://github.com/ScottDougBlain/llm-hallucination-reduction.git
cd llm-hallucination-reduction
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from src.hallucination_mitigator import HallucinationMitigator

# Initialize the framework
mitigator = HallucinationMitigator()

# Define your model response function
def your_model_response(prompt: str) -> str:
    # Replace with your LLM API call
    return your_llm_client.generate(prompt)

# Apply framework to reduce hallucinations
query = "What are the specific compliance requirements in Section 8.4 of the EU AI Act?"
result = mitigator.apply_framework(query, your_model_response)

# Compare baseline vs framework-guided responses
print("Baseline Response:", result["baseline_response"])
print("Framework Response:", result["framework_response"])
print("Improvement Metrics:", result["improvement_metrics"])
```

### Running the Benchmark Suite

```bash
# Run all 7 validation test cases
python -m src.benchmark_suite --output results.json

# Run specific test case
python -m src.benchmark_suite --test-id SZ_REFERENCES

# List all available test cases
python -m src.benchmark_suite --list-tests
```

## Proof-of-Concept Evaluation

### Test Cases

Our initial proof-of-concept used 7 test cases designed to trigger common hallucination patterns:

1. **Academic References** (`SZ_REFERENCES`): Recent citations with date constraints
2. **Pharmacological Data** (`DRUG_BINDING`): Precise binding affinity values
3. **Legal Requirements** (`EU_AI_ACT`): Specific regulatory section details
4. **Organization Metrics** (`AI_LEADERBOARD`): Institution-specific methodologies
5. **Quantitative Thresholds** (`TRUTHFULNESS_METRICS`): Specific measurement criteria
6. **Attributed Quotes** (`AI_QUOTES`): Quotations from prominent figures
7. **Article Suggestions** (`PUBLIC_PERCEPTION`): Specific publication recommendations

### Methodology

- **Model**: Claude 3.7 with thinking enabled, tool-use disabled
- **Evaluation**: Manual binary classification (hallucination present/absent)
- **Metrics**: Hallucination rate, uncertainty acknowledgment, response time
- **Sample Size**: 7 queries (insufficient for statistical significance)
- **Limitations**: No inter-rater reliability, small sample, single model tested

### Observed Patterns

In our limited testing, we observed:

- **Uncertainty acknowledgment**: Framework encouraged more explicit uncertainty statements
- **Evidence tagging**: Structure promoted citation of sources (though accuracy not verified)
- **Response length**: Framework responses were generally longer and more structured

**Caveat**: These observations are anecdotal from a small proof-of-concept. Rigorous evaluation would require larger samples, multiple evaluators, and statistical analysis.

## Technical Details

### Framework Components

#### `HallucinationMitigator`
Core class implementing the five-step metacognitive process.

```python
# Analyze any response for hallucination risk
analysis = mitigator.analyze_response(response_text)
print(f"Risk Score: {analysis['metrics']['hallucination_risk_score']}")
print(f"Evidence Coverage: {analysis['metrics']['evidence_tag_coverage']}")
```

#### `BenchmarkSuite`
Comprehensive evaluation suite with the original 7 validation test cases.

```python
# Run complete benchmark
suite = BenchmarkSuite()
results = suite.run_benchmark(model_function)
print(f"Hallucination Reduction: {results['statistics']['hallucination_reduction']:.1%}")
```

### Evidence Classification

The framework automatically extracts and classifies evidence claims:

- **URL References**: Web-verifiable sources
- **Academic Citations**: DOI/PMID tagged papers
- **Memory Claims**: Specific recalled information
- **No Evidence**: Explicitly marked unverifiable claims

### Confidence Calibration

Automatic confidence inference from linguistic markers:

- **High confidence**: "definitely", "established", "proven"
- **Low confidence**: "might", "possibly", "uncertain"
- **Medium confidence**: Default for neutral language

## API Integration Examples

### OpenAI Integration

```python
import openai
from src.hallucination_mitigator import HallucinationMitigator

client = openai.OpenAI(api_key="your-api-key")
mitigator = HallucinationMitigator()

def openai_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

result = mitigator.apply_framework("Your query here", openai_response)
```

### Anthropic Claude Integration

```python
import anthropic
from src.hallucination_mitigator import HallucinationMitigator

client = anthropic.Anthropic(api_key="your-api-key")
mitigator = HallucinationMitigator()

def claude_response(prompt: str) -> str:
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

result = mitigator.apply_framework("Your query here", claude_response)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_hallucination_mitigator.py -v
pytest tests/test_benchmark_suite.py -v

# Run tests with detailed output
pytest tests/ -v --tb=short
```

### Test Coverage

- **Unit Tests**: Core functionality and edge cases
- **Integration Tests**: End-to-end framework application
- **Benchmark Tests**: Validation suite functionality
- **Mock Tests**: API integration without external dependencies

## Configuration

### Environment Variables

```bash
# Optional: For API integration examples
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: Custom confidence thresholds
export HALLUCINATION_HIGH_CONFIDENCE_THRESHOLD=0.9
export HALLUCINATION_LOW_CONFIDENCE_THRESHOLD=0.6
```

### Custom Meta-Prompt

Modify the meta-prompt for specific domains:

```python
mitigator = HallucinationMitigator()
# Customize for medical domain
mitigator.meta_prompt = mitigator.meta_prompt + "\nAdditional medical safety considerations..."
```

## Related Work & Citations

This framework builds upon extensive research in cognitive psychology and AI safety:

### Cognitive Psychology Foundation

- **Apophenia Research**: Pattern recognition biases and false positive tendencies
- **CBT Principles**: Reality testing and metacognitive awareness techniques
- **Clinical Applications**: Delusion reduction in psychotic disorders

### AI Safety Applications

- **Truthfulness Research**: Alignment with factual accuracy objectives
- **Uncertainty Quantification**: Calibrated confidence in AI systems
- **Hallucination Detection**: Automated identification of fabricated content

### Framework Grounded In

This framework draws on clinical psychology principles from CBT and metacognitive therapy research, including:

- **Cognitive Behavioral Therapy (CBT)**: Reality-testing techniques for correcting cognitive distortions
- **Metacognitive Therapy**: Self-monitoring and awareness strategies for delusional thinking
- **Apophenia Research**: Understanding false pattern recognition and false positive bias (Blain et al., 2020)
- **Signal Detection Theory**: Frameworks for distinguishing signal from noise in cognition

## Limitations and Caveats

### Current Limitations

1. **Small Sample Size**: Only 7 test cases, insufficient for statistical validity
2. **Single Model Testing**: Evaluated only on Claude 3.7, generalization unknown
3. **Subjective Evaluation**: Manual classification without inter-rater reliability
4. **No External Validation**: Evidence tags not verified against actual sources
5. **Prompt Engineering**: Essentially sophisticated prompt engineering, not a fundamental solution
6. **Computational Overhead**: ~8% increase in response time
7. **Format Dependency**: Relies on specific tag formats that models may not consistently follow

### What This Framework Actually Is

- **A structured prompting technique** for encouraging more careful responses
- **A demonstration** of how metacognitive strategies might be implemented
- **A starting point** for research into hallucination mitigation
- **Open source code** that others can build upon and improve

### What This Framework Is NOT

- **Not a proven solution** to LLM hallucinations
- **Not validated** through rigorous scientific study
- **Not a clinical intervention** despite psychological inspiration
- **Not a guarantee** of factual accuracy

## Contributing

We welcome contributions that enhance the framework's effectiveness:

1. **Additional Test Cases**: Domain-specific hallucination scenarios
2. **Evaluation Metrics**: Novel approaches to hallucination detection
3. **Integration Examples**: New LLM API integrations
4. **Performance Optimizations**: Efficiency improvements
5. **Documentation**: Expanded examples and tutorials

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/ScottDougBlain/llm-hallucination-reduction.git
cd llm-hallucination-reduction
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- **Documentation**: See code and examples for usage details
- **Issues**: [GitHub Issues](https://github.com/ScottDougBlain/llm-hallucination-reduction/issues)

---

## AI Safety Context

This framework represents a novel application of clinical psychology insights to AI alignment challenges. By understanding hallucinations as epistemic failures analogous to human cognitive biases, we can develop more robust approaches to AI truthfulness.

### Safety Implications

- **Reduced Misinformation**: Lower hallucination rates decrease spread of false information
- **Improved Calibration**: Better uncertainty acknowledgment enables appropriate trust
- **Clinical Applications**: Safer AI deployment in high-stakes medical/legal domains
- **Alignment Research**: Framework for human-like epistemic humility in AI systems

### Future Directions

- **Multi-Modal Extension**: Applying framework to vision and audio hallucinations
- **Real-Time Integration**: Low-latency implementations for production systems
- **Domain Specialization**: Customized meta-prompts for specific fields
- **Human-AI Collaboration**: Enhancing human oversight of AI uncertainty

This framework demonstrates how interdisciplinary approaches combining psychology, neuroscience, and AI safety can yield practical solutions to fundamental alignment challenges.