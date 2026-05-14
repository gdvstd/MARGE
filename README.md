# MARGE — Multi-agent ML-Reasoning Guidance Engine

> **IBM × UNSA Hackathon 2026 Best Global Hack Solution** — Clinical AI assistant that orchestrates niche tabular ML models and a medical expert sub-agent to produce sourced, evidence-grounded clinical guidance.

![Overall architecture of the MARGE (BeeAI IBM) system](docs/architecture.png)

---

## What It Does

MARGE is a clinical decision-support system designed around a single hard constraint: **the user-facing Chat Agent never produces medical claims from its own knowledge.**

A clinician or patient uploads clinical data and asks a question. MARGE:

1. **Consults a medical expert sub-agent** for clinical reasoning and differential diagnosis
2. **Delegates ML prediction to a dedicated ML Orchestrator sub-agent** which selects from a catalog of 11 tabular clinical models — each returning a prediction, confidence, and SHAP-style feature importance
3. **Forces the ML Orchestrator to self-review its own predictions** (Phase 2 self-critique) — if confidence is not credible, the ML Orchestrator names the additional features it needs to strengthen the prediction
4. **Re-consults the expert** with ML results expressed as clinical values so the expert can interpret, confirm, or flag contradictions
5. **Produces a structured clinical report** — or a structured *clinical inquiry* when the ML Orchestrator asked for more inputs — only after both ML evidence and expert reasoning are present, enforced structurally by framework middleware, not by prompting

If the expert rules out every ML catalog condition, or if models conflict irresolvably, the Chat Agent **abstains** and refers the user to a human specialist.

---

## IBM Stack

| Component | IBM Technology | Role |
|---|---|---|
| Agent orchestration | **BeeAI Framework** (IBM Research, open-source) | ReAct-style tool-use loop for the Chat Agent, ML Orchestrator, and Medical Expert sub-agents; `RequirementAgent` middleware for protocol enforcement |
| LLM backbone | **IBM Granite 3.x via watsonx.ai** | Primary model for every agent; per-role routing with fallback support |
| Cloud storage | **IBM Cloud S3** | ML datasets, knowledge docs, and reference papers stored in object storage |
| Vectorized retrieval | **IBM Cloud** — Vector DB | Knowledge docs chunked, embedded, and indexed for semantic RAG search by the Medical Expert Agent |
| ML Agent & models | **IBM Cloud** | ML Agent trains XGBoost ensemble models on each dataset, packages them with XAI explainers (SHAP), and fetches the artifacts to local for the MCP server to serve |

### Why BeeAI over LangGraph

BeeAI keeps control flow inside the LLM — each agent decides at runtime which tool to call, iterates on results, and re-plans. A hardcoded graph would require enumerating every decision branch in advance, which breaks the "drop in a new model and the orchestrator just uses it" design goal.

The trade-off (losing graph-level flow guarantees) is offset by structural enforcement via BeeAI `RequirementAgent` middleware — the Chat Agent literally cannot call `clinical_report` until both a `consult_ml_orchestrator` result and a `consult_medical_expert` result are present in the trajectory.

---

## Architecture (3-agent)

```
┌────────────────────────────────────────────────┐
│  Streamlit UI  (apps/streamlit_ui/)            │
│  • chat interface • CSV upload • session DB    │
└────────────────┬───────────────────────────────┘
                 │
   ┌─────────────▼──────────────┐
   │  Chat Agent                │  BeeAI RequirementAgent
   │  (apps/orchestrator/)      │  "ML head researcher / coordinator"
   │                            │  — never diagnoses directly,
   │                            │    no ML schema knowledge
   └─┬───────┬──────────┬───────┘
     │       │          │
   tool│   tool│  filtered│MCP (describe-only)
     │       │          │
┌────▼────┐ ┌▼──────┐  ┌▼─────────────────────┐
│Medical  │ │ML     │  │ ML MCP Server        │
│Expert   │ │Orches-│  │ • 11 XGBoost models  │
│Sub-agent│ │trator │  │   exposed as         │
│         │ │Sub-   │  │   predict_*          │
│Tavily   │ │agent  │  │ • describe_ml_       │
│web RAG  │ │       │◀─┤   features tool      │
│(scoped) │ │       │  │   (read-only, also   │
└─────────┘ └───────┘  │   visible to         │
                       │   Chat Agent)        │
                       └──────────────────────┘
                              ▲
                              │
                       ┌──────┴──────────────┐
                       │ Patient Data MCP     │
                       │ Server               │
                       │ • SQLite seed DB     │
                       │ • CSV upload         │
                       └──────────────────────┘
```

### Three-agent role split

- **Chat Agent** (user-facing): orchestrates the conversation, routes work, formats responses. Has no medical or ML schema authority of its own — every clinical statement it relays must originate from a sub-agent call in the current trajectory.
- **ML Orchestrator sub-agent**: professional ML researcher. Selects predictors from the catalog, runs them with available patient features, performs a mandatory two-phase workflow (predict → self-review). When self-review flags a prediction as "not yet credible", it emits a structured `needed_features` list naming the catalog feature names that would most increase confidence.
- **Medical Expert sub-agent**: pure clinical reasoner. No knowledge of ML predictors. Decides when to invoke its web search tool (scoped to MedlinePlus and PubMed by default); if it searches, retrieved sources are auto-attached as `Citation` objects.

### Protocol Enforcement (Structural, Not Prompted)

Two layers of safety invariants:

- **`MARGEProtocolRequirement`** — BeeAI `Requirement` wired into the Chat Agent's planning loop. `clinical_report` is hidden until at least one `consult_ml_orchestrator` and one `consult_medical_expert` result are present; `abstain` is hidden until the expert has been consulted at least once.
- **Tool surface filtering** — the Chat Agent's connection to the ML MCP server is filtered so only the read-only `describe_ml_features` tool is exposed; `predict_*` stay exclusive to the ML Orchestrator sub-agent.

The constraint is architectural — even if the system prompt were entirely removed, the Chat Agent physically cannot produce a final report without first triggering both sub-agents, and physically cannot call a predictor directly.

---

## System Components

### Chat Agent (`apps/orchestrator/`)

BeeAI `RequirementAgent` assembly. Coordinator role:

- Holds the ML catalog (dynamically injected into its system prompt at startup) so it knows which conditions can be routed to `consult_ml_orchestrator`
- Translates ML Orchestrator output (predictions + `needed_features`) into user-facing prose and structured inquiry cards
- Per-feature display metadata is sourced from the read-only `describe_ml_features` MCP tool — the Chat Agent never invents labels, units, or feature explanations
- Terminals: `clinical_report` · `request_ml_clinical_info` · `abstain` · plus the sub-agent tools `consult_ml_orchestrator` and `consult_medical_expert`

### ML Orchestrator Sub-agent (`services/ml_orchestrator_agent/`)

BeeAI `RequirementAgent` with a clinical-ML-researcher system prompt. Holds:

- Its own LLM + persistent `UnconstrainedMemory` (separate from the Chat Agent), so it remembers prior consultations within a session
- Direct access to every `predict_*` tool on the ML MCP server
- A mandatory two-phase workflow: Phase 1 runs the predictor(s), Phase 2 self-reviews the predictions for credibility and emits a machine-readable JSON tail when more features are needed for confidence

### ML MCP Server (`services/ml_mcp_server/`)

FastMCP server exposing each ML model as a self-describing tool plus the cross-cutting `describe_ml_features` documentation tool. The registry auto-discovers every non-`_` prefixed module in `models/` at startup — **adding a new clinical predictor requires one file, no other changes.**

**Registered models (11 total):**

| Tool name | Dataset | Task |
|---|---|---|
| `predict_diabetes_risk` | Pima Indians Diabetes (OpenML, n=768) | Binary: diabetic risk vs low risk |
| `predict_type2_diabetes` | Type 2 Diabetes Dataset | Binary: T2DM risk |
| `predict_breast_cancer_malignancy` | Wisconsin Diagnostic (UCI, n=569) | Binary: malignant vs benign |
| `predict_heart_disease` | Cleveland Heart Disease (UCI, n=303) | Binary: disease present vs absent |
| `predict_heart_failure` | Heart Failure Clinical Records (n=299) | Binary: death event |
| `predict_stroke` | Healthcare Stroke Dataset | Binary: stroke risk |
| `predict_hypertension` | Synthetic Clinical Dataset | Binary: hypertension |
| `predict_liver_disease` | Indian Liver Patient Dataset (ILPD, n=583) | Binary: liver disease |
| `predict_sepsis` | ICU Sepsis Records | Binary: sepsis onset |
| `predict_dengue` | Dengue Blood Panel Dataset | Binary: dengue positive |
| `predict_synthetic_mortality` | Synthetic Clinical Dataset | Binary: in-hospital mortality |

Each prediction response includes per-feature SHAP importance scores so the ML Orchestrator can quote "what drove this prediction" in its summary.

**Authored `feature_metadata`** — each model file declares `label`, `detail`, `unit`, `field_type`, and `aliases` (including Korean / English) per feature. This metadata flows automatically into the model's Pydantic input schema (`json_schema_extra`) and is the single source of truth for user-facing feature descriptions; `describe_ml_features` simply surfaces it through MCP.

**`DynamicMLAgent` factory pattern** — new models configure themselves via `AgentConfig` (feature names, artifact path, target classes, training description, feature metadata). The factory builds the Pydantic input schema dynamically, runs K-Fold XGBoost ensemble training, sets up SHAP, and serializes to `.joblib`. Init-or-train lifecycle: if the artifact exists on disk, it loads directly.

### Medical Expert Sub-agent (`services/medical_expert_agent/`)

BeeAI `RequirementAgent` with a clinical-reasoning-only system prompt. The expert:

- Has no awareness of the ML catalog — reasons in pure clinical terms (differentials, thresholds, guidelines, referral recommendations)
- Decides when to invoke `search_medical_web` (Tavily-backed); domain whitelist defaults to MedlinePlus and PubMed, configurable via `MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS`
- Capped to at most one web search per consultation; if it searches, retrieved documents are auto-attached as `Citation` objects on the response
- Returns `MedicalExpertResponse(reasoning, citations)` — the Chat Agent quotes expert reasoning into the final report

### Patient Data MCP Server (`services/patient_data_mcp_server/`)

FastMCP server exposing patient record tools (`list_patients`, `get_patient`, `update_patient`). Two source backends resolve to the same `PatientRecord` Pydantic schema:

- **SQLite seed DB** — curated sample patients for narrative-style demos
- **CSV upload adapter** — Streamlit file upload ingested in-memory per session

### LLM Provider Abstraction (`packages/llm_provider/`)

Thin wrapper over BeeAI's model adapter. Six providers supported, per-role routing, and optional `FallbackChatModel`:

| Provider | Default model | Notes |
|---|---|---|
| **watsonx.ai** (IBM) | `ibm/granite-3-8b-instruct` | Primary — IBM hackathon stack |
| Anthropic | `claude-haiku-4-5-20251001` | Fallback |
| Cerebras | `qwen-3-235b-a22b` | Free: 30 RPM / 1M tokens/day |
| NVIDIA NIM | `qwen/qwen3-next-80b-a3b-instruct` | Free credits |
| Chutes | `moonshotai/Kimi-K2.5-TEE` | Free, 256K context |
| Featherless | `moonshotai/Kimi-K2.5` | Free |

Per-provider rate-limit throttling is built in — free-tier providers with strict RPM limits (Cerebras 30 RPM, NVIDIA 40 RPM) get a shared async lock+sleep so back-to-back agent iterations stay under the quota.

---

## Layering Rules

```
apps/  →  services/  →  packages/
```

1. `apps/` depends on `services/` and `packages/`. Never the reverse.
2. `services/` depend only on `packages/`. Services are independent — `ml_mcp_server` cannot import from `medical_expert_agent`.
3. `packages/schemas/` is the only module imported everywhere.
4. The Chat Agent accesses the Medical Expert and ML Orchestrator **only** through their `consult_*` tools — never by direct import.
5. The Medical Expert never reads patient records — if context is needed, the Chat Agent includes relevant fields in the consultation payload.
6. The Chat Agent never calls `predict_*` directly — every ML prediction is mediated by the ML Orchestrator sub-agent.

---

## Project Structure

```
marge/
├── apps/
│   ├── orchestrator/          # Chat Agent — BeeAI RequirementAgent coordinator
│   │   ├── agent.py           # agent assembly + async context manager
│   │   ├── system_prompt.md   # role, medical-knowledge boundary, new flow
│   │   ├── tools/             # consult_expert, consult_ml_orchestrator,
│   │   │                      #   request_ml_clinical_info, clinical_report, abstain
│   │   ├── middleware/        # enforce_protocol.py — gates clinical_report
│   │   └── requirements/      # marge_protocol.py — BeeAI Requirement wiring
│   └── streamlit_ui/          # chat UI, CSV upload, session management
│
├── services/
│   ├── ml_mcp_server/         # FastMCP: exposes ML models + describe_ml_features
│   │   ├── models/            # one file per model (drop-in extension point)
│   │   │   ├── _base.py       # MLModel ABC
│   │   │   ├── _agent_factory.py  # DynamicMLAgent + AgentConfig factory
│   │   │   ├── diabetes_xgb.py
│   │   │   ├── breast_cancer_xgb.py
│   │   │   ├── heart_disease_xgb.py
│   │   │   └── ... (11 models total)
│   │   ├── feature_descriptions.py  # describe_ml_features implementation
│   │   ├── registry.py        # auto-discovers models/ at startup
│   │   └── artifacts/         # serialized .joblib files (gitignored)
│   ├── ml_orchestrator_agent/ # ML researcher sub-agent (BeeAI, persistent memory)
│   ├── medical_expert_agent/  # Clinical reasoner sub-agent (BeeAI)
│   └── patient_data_mcp_server/  # FastMCP: patient records (SQLite + CSV)
│
├── packages/
│   ├── schemas/               # Pydantic v2 shared types
│   │   ├── prediction.py      # Prediction, XAIScore, ModelMetadata
│   │   ├── patient.py         # PatientRecord, ClinicalFeature
│   │   ├── retrieval.py       # MedicalExpertResponse, Citation, RetrievedDocument
│   │   └── ml.py              # MLOrchestratorResponse, NeededFeature, FeatureDescription
│   ├── llm_provider/          # provider abstraction, per-role routing, throttle
│   ├── ml_training/           # offline training scripts
│   └── medical_kb/            # local RAG corpus (Chroma + sentence-transformers)
│
└── tests/
    ├── unit/                  # per-module pytest
    ├── integration/           # MCP ↔ orchestrator wiring
    └── e2e/                   # Streamlit + full-stack flows
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install core + orchestrator + UI dependencies
uv sync --all-extras

# 2. Train the ML artifacts (writes .joblib under services/ml_mcp_server/artifacts/)
uv run python -m packages.ml_training.train_breast_cancer
uv run python -m packages.ml_training.train_diabetes

# 3. Unit + integration tests
uv run pytest tests/unit tests/integration -q

# 4. Configure credentials
cp .env.example .env
# Paste your provider keys (see .env.example for the full list)

# 5. Run the Streamlit UI
uv run streamlit run apps/streamlit_ui/app.py
```

Optional extras (already included in `--all-extras`):

```bash
uv sync --extra medical-kb   # Tavily web RAG for expert citations (set TAVILY_API_KEY)
uv sync --extra dev          # ruff linter + pytest extras
```

### Provider Configuration (`.env`)

```ini
# Primary: IBM Granite via watsonx.ai
LLM_PROVIDER=watsonx
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Per-role routing (override primary per agent)
ORCHESTRATOR_PRIMARY=watsonx
ML_ORCHESTRATOR_PRIMARY=watsonx
MEDICAL_EXPERT_PRIMARY=watsonx

# Optional fallback (e.g., Cerebras free tier)
ORCHESTRATOR_FALLBACK=cerebras
CEREBRAS_API_KEY=...

# Expert web RAG
TAVILY_API_KEY=...
MARGE_WEB_RAG_MAX_RESULTS=3
MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS=medlineplus.gov,pubmed.ncbi.nlm.nih.gov
```

---

## Adding a New ML Model

1. Create `services/ml_mcp_server/models/your_model.py`
2. Instantiate `AgentConfig` with feature names, artifact path, dataset description, and `feature_metadata` (label / detail / unit / field_type / aliases per feature)
3. Subclass `DynamicMLAgent` and implement `__init__` (trigger training) + `sample_inputs()`
4. The registry auto-discovers it on next server start; the ML Orchestrator gains the predictor via MCP and the Chat Agent gains feature documentation via `describe_ml_features`

No other files need to change.

---

## Runtime Data Flow

```
User query (+ optional CSV patient data)
   │
   ▼  Streamlit session
Chat Agent (BeeAI RequirementAgent, Granite / watsonx.ai)
   │
   ├─ get_patient / update_patient  ──MCP──▶  patient_data_mcp_server
   │
   ├─ consult_medical_expert()              medical_expert_agent (BeeAI sub-agent)
   │     └─ search_medical_web()  ──Tavily──▶  MedlinePlus / PubMed (whitelisted)
   │     └─ returns MedicalExpertResponse(reasoning, citations)
   │
   ├─ consult_ml_orchestrator()             ml_orchestrator_agent (BeeAI sub-agent)
   │     ├─ Phase 1: predict_*  ──MCP──▶  ml_mcp_server  (XGBoost + SHAP)
   │     ├─ Phase 2: self-review → optional needed_features JSON tail
   │     └─ returns MLOrchestratorResponse(reasoning, needed_features?)
   │
   ├─ describe_ml_features(names=[...])  ──MCP──▶  ml_mcp_server  (read-only)
   │     └─ returns label / description / unit / field_type / aliases
   │
   ├─ consult_medical_expert()  (second pass — ML results → clinical interpretation)
   │
   └─ [RequirementAgent checks: ML ✓ + expert ✓]
      ├─ clinical_report(...)            ──▶  structured report card
      ├─ request_ml_clinical_info(...)   ──▶  structured clinical inquiry card
      └─ abstain(reason, fallback)       ──▶  scope-mismatch warning
```

If the ML Orchestrator's Phase 2 returns `needed_features` (i.e. predictions weren't credible enough), the Chat Agent forwards them to `request_ml_clinical_info`, which renders a structured inquiry card asking the user only for the specific missing model features. Free-form clarifying questions go in natural-language chat replies, not via this tool.

---

## License

Apache 2.0
