# Adaptive AI Interview Coach

A stateful and adaptive technical interview agent built with **LangGraph**, local LLMs, persistent memory, tools, MCP, human-in-the-loop review, evals, and observability.

The agent interviews users on topics such as **Machine Learning, Deep Learning, LLMs, AI Agents, and Healthcare AI**, evaluates their answers, remembers their performance, and adapts future questions accordingly.

## Architecture

```text
User
  ↓
LangGraph Interview Orchestrator
  ↓
Generate Question
  ↓
Collect Answer
  ↓
Model Router ──→ Fast / Strong Model
  ↓
Evaluator
  ↓
Human Review (when confidence is low)
  ↓
Persistent Skill Memory
  ↓
Interview Planner
  ├── harder / easier
  ├── probe / new skill
  └── coach
  ↓
Tools / MCP
  ↓
Continue or Finish
```

The system separates three main responsibilities:

- **Interviewer** — generates and adapts questions
- **Evaluator** — assesses candidate answers
- **Coach** — explains weaknesses and recommends areas to review

LangGraph coordinates the workflow and maintains interview state.

## Key Features

### Adaptive Interviewing
The next interview step is selected using the current evaluation and the candidate's previous performance.

Possible strategies include:

`harder` · `easier` · `probe` · `new_skill` · `coach`

### Multi-Model Routing
A lightweight router determines whether an answer can be evaluated by a fast model or should be escalated to a stronger model.

| Role | Model |
|---|---|
| Fast | Gemma 3 1B |
| Strong | Qwen 3 1.7B |
| Router | Qwen 2.5 1.5B |

Models currently run locally through **Ollama**.

### Persistent Memory
Interview performance is stored in **SQLite** as skill profiles containing attempts, average performance, and identified weaknesses.

This memory is used to adapt future questions.

### Structured Outputs
Important LLM decisions use **Pydantic schemas** for outputs such as:

`Evaluation` · `RoutingDecision` · `InterviewPlan` · `InterviewQuestion` · `CoachOutput`

### Tools + MCP
The agent uses tools for operations such as retrieving question banks and saving interview results.

MCP is used to explore standardized tool exposure, while guarded operations can require confirmation before execution.

### Human-in-the-Loop
Low-confidence evaluations can pause the LangGraph workflow using `interrupt()` and allow a human to review the decision before execution continues.

### Observability
**Langfuse** traces the interview workflow, including LLM calls, routing, latency, inputs/outputs, and failures.

## Evals & Reliability

V8 introduced behavioral evals for the **Evaluator** and **Interview Planner**.

Initial baseline:

| Component | Result |
|---|---:|
| Evaluator | 3/5 |
| Planner | 3/5 |

The evals exposed two current limitations: the evaluator can be too lenient between `poor` and `partial`, while the planner can overuse `probe`.

These results are intentionally kept as a baseline rather than tuning specifically to the small eval set.

The project also includes production-oriented safeguards:

- LLM timeouts
- bounded retries
- model fallback
- structured-output validation
- database rollback
- graph recursion limits
- guarded tool execution
- human review

Failure scenarios including model, Ollama, database, and graph failures were deliberately tested.

## Project Evolution

The project was intentionally built incrementally, introducing one major agent concept at a time.

| Version | Focus |
|---|---|
| V1 | Basic LangChain interviewer |
| V2 | LangGraph + state |
| V3 | Multi-model architecture |
| V4 | Model routing |
| V5 | Tools + MCP + tool guard |
| V6 | Persistent memory |
| V7 | Multi-agent roles + HITL |
| V8 | Evals + observability + production safeguards |

```text
State
  → model routing
  → tools
  → MCP
  → memory
  → adaptive orchestration
  → HITL
  → evals
  → observability
  → production safeguards
```

## Tech Stack

**Python · LangChain · LangGraph · Ollama · Qwen · Gemma · Pydantic · SQLite · SQLAlchemy · MCP · Langfuse · Pytest · Ruff**

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the local models

```bash
ollama pull gemma3:1b
ollama pull qwen3:1.7b
ollama pull qwen2.5:1.5b
```

### 3. Configure Langfuse

Create a `.env` file:

```env
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 4. Run the interview

```bash
python main.py
```

## Testing

Run the deterministic test suite:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

LLM behavioral evals are kept separately from deterministic tests because they measure model quality and may vary between runs.

## Current Limitations

This is an experimental engineering project rather than a production interview platform. It currently relies on small local models, a small evaluation dataset, SQLite memory, and an in-memory LangGraph checkpointer.

## Future Improvements

- Compare local routing against hosted LLM APIs
- Expand the behavioral evaluation dataset
- Improve evaluator and planner calibration
- Integrate the production Jev decision layer
- Add persistent LangGraph checkpointing
- Add automated regression evals
- **Containerised deployment**
- Add a lightweight web or voice interface

## Purpose

This project was built to explore the main engineering concepts behind modern agentic systems while deliberately keeping the implementation compact enough to understand each component and its role.