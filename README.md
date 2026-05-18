# control-layer
A production-grade control layer that sits between your application logic and any LLM — input validation, schema enforcement, circuit breaking, targeted retry, and audit logging in one composable pipeline.


![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/deps-tiktoken%20%7C%20tenacity%20%7C%20pydantic%20%7C%20structlog-lightgrey)

Most LLM integrations stop at: write a prompt, call the model, use the response. This
library handles what prompt engineering cannot — enforcing what the model actually returns,
blocking what should never reach it, and recovering cleanly when things break.

Read the full write-up on Towards Data Science →
**Prompt Engineering Failed in Production — I Built the Control Layer That Actually Works**

---

## What It Does

```
User Input
    |
[1] InputGuard          -- injection detection (20 patterns), length check, sanitization
    |
[2] CircuitBreaker      -- stops hammering a failing LLM backend
    |
[3] TokenBudget         -- tiktoken-accurate slot allocation, priority order
[4] PromptBuilder       -- assembles prompt within budget, injects constraints
    |
[5] LLMCaller           -- enforces hard timeout on every call
    |
[6] ResponseValidator   -- JSON schema, length bounds, forbidden phrases, quality score
    | [failed?]
[7] RetryEngine         -- targeted prompt mutation per failure mode, jittered backoff
    | [exhausted?]
[8] FallbackRouter      -- cached response, template, or escalation chain
    |
    AuditLogger         -- every attempt written to JSONL, thread-safe, persistent
    |
ControlPacket           -- response, attempts, latency, score, audit_id
```

| Component | Job |
|---|---|
| InputGuard | Blocks injection attempts and oversized input before any LLM call |
| CircuitBreaker | Opens after N consecutive failures; rejects calls instantly during recovery |
| TokenBudget | tiktoken-accurate slot-based allocator; prevents silent overflow |
| PromptBuilder | Assembles prompt in priority order with hard constraints injected structurally |
| LLMCaller | Wraps any callable LLM with thread-based timeout enforcement |
| ResponseValidator | Validates JSON structure, required keys, length, forbidden phrases |
| RetryEngine | Maps each failure mode to a targeted mutation hint; jittered exponential backoff |
| FallbackRouter | Registered fallback chain; first non-empty response wins |
| AuditLogger | Thread-safe JSONL audit log; P50/P90/P99 latency stats; failure distribution |

---

## Installation

```bash
git clone https://github.com/Emmimal/control-layer.git
cd control-layer
pip install tiktoken tenacity pydantic structlog   # required
pip install pytest                                  # optional — for running tests
```

No ML dependencies. No GPU required. All functionality runs on the Python standard library
plus the four packages above.

---

## Quick Start

```python
from control_layer import ControlLayer, ControlLayerConfig, ResponseSchema

# Define your output contract
schema = ResponseSchema(
    must_be_json=True,
    required_keys=["summary", "confidence"],
    max_length=400,
    forbidden_phrases=["I cannot", "As an AI"],
)

# Configure the layer
config = ControlLayerConfig(
    total_tokens=800,
    max_attempts=3,
    timeout_seconds=30.0,
    cb_failure_threshold=5,
    cb_recovery_seconds=30.0,
)

# Swap in any LLM callable — OpenAI, Anthropic, local model, mock
def your_llm_call(prompt: str) -> str:
    ...

layer = ControlLayer(
    llm_fn=your_llm_call,
    system_prompt="You are a structured research assistant.",
    schema=schema,
    config=config,
)

# Register fallbacks — called in order when retries exhaust
layer.register_fallback(
    "cache",
    lambda q: '{"summary": "Cached response.", "confidence": 0.5}',
)

# Run
packet = layer.run(
    user_input="How does token budget allocation work?",
    constraints=[
        "Return only valid JSON.",
        "Include 'summary' and 'confidence' keys.",
        "No markdown fencing.",
    ],
    context=retrieved_documents,   # optional RAG context
)

print(packet.response)            # final response
print(packet.validation.passed)   # True / False
print(packet.attempts)            # 1, 2, or 3
print(packet.total_latency_ms)    # end-to-end latency
print(packet.audit_id)            # ties all log lines to this request
```

---

## Running the Demos

Five runnable demos covering every failure mode and recovery path. No API key required.
The `MockLLM` simulates realistic failure behavior at a configurable rate.

```bash
python demo.py
```

| Demo | What It Shows |
|---|---|
| 1 | Input guard blocking 7 of 8 inputs — injection, empty, oversized |
| 2 | Schema enforcement with retry — 75% first-attempt failure rate, mutation hints |
| 3 | Constraint violation recovery — length and forbidden phrase, 3 attempts |
| 4 | Fallback router — exhausted retries route to cached response |
| 5 | Benchmark — naive 0% pass rate vs control layer 100%, latency breakdown |

Running Demo 5 also generates `control_layer_benchmark.png` — a 6-panel benchmark figure
showing pass rate, failure mode distribution, retry distribution, latency percentiles,
token budget allocation, and quality score histogram.

---

## Running the Tests

```bash
pytest tests/ -v
```

```
TestInputGuard               14 tests   PASSED
TestTokenBudget               5 tests   PASSED
TestPromptBuilder             6 tests   PASSED
TestResponseValidator        10 tests   PASSED
TestCircuitBreaker            5 tests   PASSED
TestRetryEngine               6 tests   PASSED
TestFallbackRouter            4 tests   PASSED
TestLLMCaller                 2 tests   PASSED
TestAuditLogger               5 tests   PASSED
TestControlLayerIntegration   8 tests   PASSED
TestPydanticConfig            4 tests   PASSED

69 passed in 1.19s
```

Every component is tested in isolation. Integration tests cover the full orchestration
path: first-attempt success, retry on schema violation, fallback after exhausted retries,
circuit breaker rejection after consecutive timeouts, and Pydantic config validation errors.

---

## Configuration Reference

```python
ControlLayerConfig(
    # Token budget
    total_tokens=800,              # Total token budget for prompt assembly
    model_name="cl100k_base",      # tiktoken encoding name

    # Input validation
    max_input_chars=2000,          # Hard limit on user input length

    # LLM call
    timeout_seconds=30.0,          # Hard timeout per LLM call

    # Retry
    max_attempts=3,                # Maximum retry attempts per request
    base_delay_ms=50.0,            # Base exponential backoff delay
    max_delay_ms=2000.0,           # Maximum backoff delay
    jitter_ms=25.0,                # Random jitter added to each delay

    # Circuit breaker
    cb_failure_threshold=5,        # Consecutive failures before opening
    cb_recovery_seconds=30.0,      # Seconds before attempting recovery

    # Audit
    audit_log_path="audit.jsonl",  # JSONL audit log path
)
```

```python
ResponseSchema(
    must_be_json=False,            # Require valid JSON response
    required_keys=[],              # Keys that must appear in JSON output
    max_length=None,               # Maximum response length in characters
    min_length=None,               # Minimum response length in characters
    forbidden_phrases=[],          # Phrases that must not appear in response
    must_contain=[],               # Phrases that must appear (used for quality score)
)
```

---

## Swapping the LLM

The `llm_fn` parameter accepts any callable that takes a `str` and returns a `str`.

```python
# OpenAI
import openai
client = openai.OpenAI()

def openai_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

layer = ControlLayer(llm_fn=openai_call, ...)

# Anthropic
import anthropic
client = anthropic.Anthropic()

def claude_call(prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

layer = ControlLayer(llm_fn=claude_call, ...)

# Any local model
layer = ControlLayer(llm_fn=lambda prompt: your_local_model.generate(prompt), ...)
```

---

## Project Structure

```
control-layer/
├── control_layer.py          # All eight components + ControlLayer orchestrator
├── demo.py                   # Five runnable demos + benchmark charts
├── tests/
│   └── test_control_layer.py # 69 tests across all components
├── audit.jsonl               # Generated on first run (append-only audit log)
├── control_layer_benchmark.png  # Generated by demo.py
└── README.md
```

---

## Benchmark

Measured on Python 3.12.6, Windows 11, CPU only, no GPU.
Ten structured output queries, 55% first-attempt failure rate.

| Metric | Naive | Control Layer |
|---|---|---|
| Pass rate | 0% | 100% |
| Min latency (ms) | 37.3 | 46.2 |
| Median latency (ms) | 43.3 | 143.5 |
| Mean latency (ms) | 42.9 | 139.8 |
| P90 latency (ms) | 45.6 | 168.0 |
| Max latency (ms) | 48.4 | 281.9 |
| Resolved on attempt 1 | N/A | 2 |
| Resolved on attempt 2 | N/A | 7 |
| Resolved on attempt 3+ | N/A | 1 |

Component overhead (excluding LLM call):

| Operation | Latency | Notes |
|---|---|---|
| InputGuard validation | ~0.2ms | 20 regex patterns |
| tiktoken count (100 tokens) | ~0.8ms | Encoding lookup |
| PromptBuilder.build() | ~1.1ms | Budget allocation + assembly |
| ResponseValidator.validate() | ~0.3ms | JSON parse + rule checks |
| CircuitBreaker.is_open() | ~0.05ms | Lock acquire + state check |
| AuditLogger.log() | ~0.4ms | Lock + file append |
| Total non-LLM overhead | ~2.9ms | Per request |

The LLM call dominates every other number. The control layer adds under 3ms of overhead
per request, which is within the variance of a single network round-trip.

---

## When to Use This

Worth it when you have:

- LLM responses that drive downstream code — JSON parsed programmatically, data written
  to a database, outputs shown to users without human review
- User input passed to an LLM without a validation layer in between
- Structured output requirements the model violates intermittently
- Production systems where a LLM outage would block threads or hang requests

Skip it when you have:

- Single-turn, low-stakes use cases where a bad response is displayed and discarded
- Hard latency requirements under 50ms — retry delays alone can exceed this
- A chatbot where the user sees the raw model output and can judge it themselves

---

## Known Limitations

**Injection patterns are not exhaustive.** Twenty patterns cover the OWASP LLM Top 10
attack taxonomy. Adversarial prompts crafted to avoid known patterns will pass. Combine
with embedding-based anomaly detection for high-risk deployments.

**Circuit breaker state is in-process only.** A restart resets the circuit to CLOSED
regardless of backend status. For multi-instance deployments, share circuit state via
Redis or a similar low-latency store.

**No streaming support.** The `LLMCaller` collects the full response before validation.
Streaming APIs require partial validation heuristics or full response buffering — neither
is implemented.

**Quality score uses phrase matching, not semantic similarity.** `must_contain` checks
exact string presence. A response that paraphrases a required concept without using the
exact phrase scores zero. Swap in an embedding-based scorer for higher precision.

**AuditLogger grows unbounded.** The JSONL file appends on every call. In production,
ship it to object storage on a rolling basis and rotate locally.

---

## Related

- ## Related

**Same series — production layers for LLM systems:**

- [context-engine](https://github.com/Emmimal/context-engine) — retrieval,
  re-ranking, memory decay, and token budget control for RAG systems. The
  control layer handles what the model returns. The context engine handles
  what it receives. They compose.

- [RAG Is Blind to Time — I Built a Temporal Layer to Fix It in Production](https://towardsdatascience.com/rag-is-blind-to-time-i-built-a-temporal-layer-to-fix-it-in-production/)
  — temporal awareness layer for RAG systems that treats time as a first-class
  retrieval signal.

- [LLM Evals Are Based on Vibes — I Built the Missing Layer That Decides What Ships](https://towardsdatascience.com/llm-evals-are-based-on-vibes-i-built-the-missing-layer-that-decides-what-ships/)
  — evaluation layer that replaces gut-feel shipping decisions with measurable
  output quality gates.

- [PyTorch NaNs Are Silent Killers — I Built a 3ms Hook to Catch Them at the Exact Layer](https://towardsdatascience.com/pytorch-nans-are-silent-killers-i-built-a-3ms-hook-to-catch-them-at-the-exact-layer/)
  — lightweight hook that catches NaN propagation at the exact layer it
  originates, in under 3ms overhead.

---

## License

MIT
