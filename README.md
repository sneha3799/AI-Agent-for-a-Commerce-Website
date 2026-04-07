# Commerce AI Agent

An AI-powered shopping assistant that handles general conversation, text-based product recommendations, and image-based product search through a single unified agent.

> **Reference:** [Amazon Rufus](https://www.aboutamazon.com/news/retail/amazon-rufus)

---

## Table of Contents

1. [Features](#features)
2. [Screenshots](#screenshots)
3. [Architecture Overview](#architecture-overview)
4. [Technology Stack](#technology-stack)
5. [Technology Choices and Trade-offs](#technology-choices-and-trade-offs)
6. [Constraint-Driven Design](#constraint-driven-design)
7. [The Latency vs Quality vs Cost Triangle](#the-latency-vs-quality-vs-cost-triangle)
8. [Failure Modes and Guardrails](#failure-modes-and-guardrails)
9. [Observability with Arize Phoenix](#observability-with-arize-phoenix)
10. [Input Sanitization](#input-sanitization)
11. [Getting Started](#getting-started)
12. [Production Roadmap](#production-roadmap)
13. [Key Architectural Decisions Summary](#key-architectural-decisions-summary)

---

## Features

A single agent handles all three use cases through GPT-4o's native tool-calling:

- **General conversation** — "What's your name?", "What can you do?" — The agent responds directly to open-ended questions without invoking any retrieval tool.

![General chat — agent responding to "what is your name?"](UI_images/general_chat_latest.png)

- **Text-based product recommendation** — "Recommend me blue jeans for men" — the agent calls `product_recommendation`, which embeds the query with CLIP and searches pgvector for similar products. 
A natural language query is embedded with CLIP and matched against the product catalog via pgvector cosine similarity.

![Text search — "blue denim jacket" returns three product recommendations](UI_images/product_recommendation_latest.png)

- **Image-based product search** — User uploads a product image — GPT-4o sees the image via its vision capability and calls `image_product_search`, which embeds the image with CLIP and searches pgvector for visually similar products.

![Image search — uploaded beige trousers matched to five similar products](UI_images/image_product_search_latest.png)

---

## Architecture Overview

![High-level architecture diagram](UI_images/architecture.png)

**How the agent loop works:**

1. User submits a text query (and optionally an image) via the Flask UI
2. The query (and image as base64, if provided) is sent to GPT-4o along with tool definitions
3. GPT-4o decides: respond directly (general chat) or call a tool (product search)
4. If a tool is called, the corresponding Python function executes: CLIP embeds the input, pgvector returns the nearest products
5. Tool results are sent back to GPT-4o, which generates a natural language response
6. The response is rendered in the browser

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM / Orchestrator | GPT-4o (OpenAI Chat Completions API) | Intent routing via function calling, vision for image input, response generation |
| Embedding model | CLIP ViT-B/32 (via open_clip) | Multimodal embedding — text and images into shared 512-dim vector space |
| Vector search | pgvector (PostgreSQL extension) | Cosine similarity search on product embeddings |
| Database | PostgreSQL | Product catalog (44K products from Myntra dataset) + embeddings in one table |
| Backend | Flask | API routes, file upload handling, template rendering |
| Frontend | Bootstrap 5 + vanilla JS | Search UI with text input and image upload |
| Dataset | Myntra Fashion Products (Kaggle) | 44,419 products with images, categories, and attributes |

---

## Technology Choices and Trade-offs

### Why pgvector instead of a dedicated vector database (Pinecone, FAISS)

Product data and embeddings live in the same database, same table, same query. This eliminates sync bugs between a separate vector store and a relational database. A single SQL query combines vector similarity search with metadata filtering:

```sql
SELECT id, product_display_name
FROM products
WHERE master_category = 'Footwear' AND base_colour = 'Red'
ORDER BY embedding <=> query_vector
LIMIT 5;
```

With FAISS, you'd retrieve top-50 by vector similarity, then filter in Python for category/color, hoping enough results survive. With Pinecone, you'd manage two services and keep them in sync. pgvector gives you one source of truth.

**Trade-off:** pgvector is slower than FAISS for pure vector search at 10M+ vectors. At 44K products, this is irrelevant — queries return in milliseconds. For production at scale, Pinecone or Milvus would be the migration path.

### Why CLIP (open_clip) for embeddings

CLIP embeds text and images into the same 512-dimensional vector space. "Red running shoes" as text and a photo of red running shoes land near each other. This means one embedding column in Postgres handles both text search and image search — no separate pipelines.

**Trade-off:** CLIP's text understanding is shallower than dedicated text embedding models (like OpenAI text-embedding-3-large). For pure text queries, a dedicated model would give 10-15% better retrieval accuracy. The mitigation: CLIP's image embeddings capture visual information (color, style, shape) that short product names miss, and the LLM can ask clarifying questions when results seem off.

### Why OpenAI function calling instead of a framework (LangChain, Strands)

Three tools, one agent, one process. OpenAI's function calling handles tool routing natively — define tool schemas as JSON, GPT-4o decides which to call, we execute and feed results back. No framework abstraction needed.

The tool functions are structured as standalone Python functions, so they can be wrapped with a `@tool` decorator (Strands) or exposed as MCP tools later without rewriting any logic.

**Trade-off:** No built-in memory, guardrails, or observability. These are planned for the production version via AWS Bedrock (see Production Roadmap).

### Why Postgres over MongoDB

Product catalogs are relational and structured — every product has the same schema (name, price, category, color, etc.). Postgres provides ACID transactions for inventory updates, efficient SQL filtering alongside vector search (via pgvector), and a mature ecosystem. MongoDB's schema flexibility is a liability when your data is inherently structured.

---

## Constraint-Driven Design

Production systems are designed from constraints inward, not from features outward:

| Constraint | Value | Implication |
|---|---|---|
| Latency SLO | P95 < 2s text, < 3s image | Can't do multi-hop retrieval or multi-agent orchestration |
| Cost ceiling | Approx $500-1,000/month at moderate traffic | Can't route 100% of traffic to GPT-4o at scale |
| Quality floor | Recommendations must come from the catalog | Must ground in retrieval, can't rely on LLM hallucinating product names |
| Catalog size | 44K products | pgvector handles this comfortably, no need for distributed search |
| Modality | Text + image input | Need a multimodal embedding model (rules out text-only embedders) |

---

## The Latency vs Quality vs Cost Triangle

You pick two. You design mitigation for the third.

**Our current choice: Quality + Latency (demo phase).** GPT-4o handles all traffic. Quality is high, latency is good, but cost is uncontrolled at scale.

**Production target: Quality + Cost, with latency mitigation.** Dual-model routing — a lightweight model (Mistral-7B / Llama-3 via AWS Bedrock) handles 80% of traffic (simple queries, general chat), while GPT-4o handles the remaining 20% (complex queries, image reasoning).

The math at 200 requests/hr:
- 80% on small model: approx $0.002/request, $9.60/day
- 20% on GPT-4o: approx $0.03/request, $28.80/day
- Total: approx $38.40/day (approx $1,150/month)
- Compare to 100% GPT-4o: $144/day (approx $4,320/month)

The routing layer adds approx 50ms but saves roughly $3,000/month.

---

## Failure Modes and Guardrails

### Currently implemented

- **Tool-call loop limit:** The agent loop runs a maximum number of tool-calling rounds to prevent runaway API costs from malformed queries.
- **Graceful error handling:** Database connection failures and tool execution errors are caught and surfaced as user-friendly flash messages rather than crashing the app.
- **Input validation:** Empty queries are rejected before reaching the agent.
- **Prompt injection sanitization:** `sanitize_input()` strips control characters, enforces a 500-character length cap, and pattern-matches against injection trigger phrases (instruction overrides, persona hijacks, prompt exfiltration, template injection, HTML/JS injection) before any user input reaches the LLM. See [Input Sanitization](#input-sanitization) for details.
- **Observability:** All LLM calls are traced end-to-end via Arize Phoenix — latency, token cost, input messages, and tool call arguments are captured automatically per request. See [Observability with Arize Phoenix](#observability-with-arize-phoenix) for details.

### Planned for production (via AWS Bedrock Guardrails)

- **Cost guardrails:** Per-request token cap (3,000 input + 1,000 output), per-user daily cap (15,000 tokens/day). If either cap is hit, return a downgraded response rather than failing silently.
- **Content filtering:** PII detection and denied topic filtering. Bedrock Guardrails provides this as configuration, complementing the existing code-level injection sanitization.
- **Retrieval quality:** Confidence threshold on similarity scores — if the top result's score is below 0.3, ask the user to refine their query rather than showing irrelevant products.
- **LLM fallback chain:** GPT-4o failure triggers fallback to Claude Sonnet, then to a smaller model with a quality degradation warning.
- **Extended observability:** Retrieval latency (P50/P95/P99), tool-call distribution, error rates, and per-user token budgets. CloudWatch dashboards with alerts on daily spend exceeding 120% of budget.

---

## Observability with Arize Phoenix

Every LLM call in the agent is automatically traced using [Arize Phoenix](https://phoenix.arize.com/) via the OpenTelemetry-based `OpenAIInstrumentor`. No manual span management is needed — registering the instrumentor at startup patches the OpenAI client so both turns of the agent loop (tool-dispatch and final answer) are captured as linked spans.

```python
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    project_name=os.getenv("PHOENIX_PROJECT_NAME", "ecommerce-ai-agent"),
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

### What gets captured per request

| Field | Example |
|---|---|
| Span kind | `LLM` (ChatCompletion) |
| Model | `gpt-4o-2024-08-06` |
| Input messages | System prompt, user query, tool results |
| Tool calls | `product_recommendation({"query": "red denim jacket"})` |
| Output | Final natural language response |
| Latency | End-to-end per span |
| Cost | Token-based cost estimate |

A two-turn agent request (tool-dispatch + final answer) produces **two linked ChatCompletion spans** within a single trace, so you can see exactly what was passed to each turn and how long each took independently.

### Spans view — project dashboard

The Spans tab shows all LLM calls across the project with status, latency P50/P99, cost, and input/output previews at a glance.

![Arize Phoenix spans view — two ChatCompletion spans for a "red denim jacket" query](UI_images/arize-phoenix.png)

### Trace detail — tool call inspection

Clicking into a trace reveals the full input message history, the tool call the model decided to make (including the parsed arguments), and the tool result fed back for the second turn.

![Phoenix trace detail — system prompt, user query, and product_recommendation tool call visible](UI_images/traces.png)

### Running Phoenix locally

```bash
pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-openai
python -m phoenix.server.main serve   # starts on http://localhost:6006
```

Then set in `.env`:

```
PHOENIX_PROJECT_NAME=ecommerce-agent
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
```

---

## Input Sanitization

User queries are sanitized in `sanitize_input()` before they reach the agent, preventing prompt injection and bounding token cost.

**Four-step pipeline:**

1. **Strip whitespace** — trim leading/trailing whitespace.
2. **Remove control characters** — strip ASCII control bytes (`0x00–0x1F`, `0x7F`) except tab and newline. These are invisible to users but can confuse tokenizers and are never present in legitimate product queries.
3. **Length cap** — reject queries over 500 characters. Unusually long inputs are a common vector for burying injected instructions after legitimate-looking text.
4. **Injection pattern matching** — scan against a compiled regex blocklist:

| Pattern category | Example trigger phrases |
|---|---|
| Instruction override | `ignore previous instructions`, `disregard all prior instructions` |
| Persona hijack | `you are now a ...`, `act as a ...`, `pretend to be ...` |
| Prompt exfiltration | `reveal your prompt`, `print your system prompt` |
| Rule bypass | `do not follow your prompt`, `override your guidelines` |
| Template injection | `{{ ... }}`, `${ ... }` |
| HTML/JS injection | `<script>` tags |

**On a blocked input**, the raw query is written to `app.logger.warning` (visible in your server logs and Phoenix traces) but is **not echoed back** to the user — echoing rejected content can help an attacker tune their payload.

```python
query, error = sanitize_input(raw_query)
if error:
    flash(error, "danger")
    return redirect(url_for("index"))
```

The sanitized string is what gets passed to `run_agent()` and, by extension, what appears in Phoenix traces — so traces always reflect the cleaned input, not raw user content.

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- OpenAI API key

### Database setup

```bash
# Using Docker (recommended)
docker run --name commerce-db \
  -e POSTGRES_PASSWORD=localdev \
  -e POSTGRES_DB=myntradataset \
  -p 5432:5432 -d pgvector/pgvector:pg16
```

### Dataset

Download the [Myntra Fashion Products dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle. Place the CSV in `myntradataset/styles.csv` and product images in `static/images/`.

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/palona-ai-agent.git
cd palona-ai-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys and database URL
```

### Environment variables (.env)

```
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://postgres:localdev@localhost/myntradataset
URL=host=localhost dbname=myntradataset user=postgres password=localdev
```

### Ingest product catalog

Generates CLIP embeddings for all 44K product images and loads them into PostgreSQL with pgvector. Takes approximately 18-20 minutes on CPU:

```bash
python create_db.py
```

### Run the agent

```bash
python app.py
```

Visit `http://localhost:8000`.

### Project structure

```
palona_ai_agent/
├── app.py                 # Flask app setup + routes only
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py    # run_agent() + tool definitions
│   └── tools.py           # product_recommendation(), image_product_search()
├── retrieval/
│   ├── __init__.py
│   └── embedder.py        # generate_embeddings(), CLIP model loading
├── guardrails/
│   ├── __init__.py
│   └── sanitizer.py       # sanitize_input(), injection patterns
├── observability/
│   ├── __init__.py
│   └── traces.py       # tracer_provider, instrumentor initialization
├── create_db.py           # Database ingestion script
├── static/
├── templates/
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Production Roadmap

The current implementation uses OpenAI directly. The production version swaps the orchestration layer to AWS Bedrock while keeping the same tool functions, CLIP embeddings, and pgvector database.

| Layer | Current (Demo) | Production Target |
|---|---|---|
| LLM | GPT-4o (OpenAI API) | GPT-4o (20%) + small model (80%) via AWS Bedrock |
| Agent framework | OpenAI function calling | Strands Agents SDK |
| Guardrails | Code-level (input validation, error handling) | Bedrock Guardrails (content filtering, PII, grounding) |
| Memory | Stateless (per request) | Bedrock AgentCore Memory (short-term + long-term) |
| Deployment | Local Flask | AWS Bedrock AgentCore Runtime (serverless) |
| Vector search | pgvector | Same (or Pinecone at scale) |
| Embedding | CLIP ViT-B/32 | Same |

The migration is clean because the tool functions (`product_recommendation`, `image_product_search`) stay identical. Only the orchestration layer changes — who decides which tool to call.

---

## Key Architectural Decisions Summary

| Decision | Choice | Constraint | Trade-off |
|---|---|---|---|
| Agent pattern | Single agent with tool-calling | Latency budget — can't afford multi-agent orchestration | Losing modularity, gaining approx 300ms |
| Vector store | pgvector (inside Postgres) | Product data + embeddings in one place, SQL filtering | Slower than FAISS at 10M+ vectors, fine at 44K |
| Database | PostgreSQL | Relational product data, ACID guarantees, pgvector support | Less "flexible" than MongoDB, but flexibility is a liability for structured catalog data |
| Embedding | CLIP ViT-B/32 | Must support both text and image in one embedding space | Weaker text retrieval vs dedicated text embedder (10-15% gap), mitigated by visual richness |
| LLM (demo) | GPT-4o for all traffic | Simplicity, fast iteration | Cost uncontrolled at scale — production adds dual-model routing |
| LLM (production) | GPT-4o (20%) + small model (80%) | Cost ceiling approx $1,500/month | 50ms routing overhead, slight quality reduction on simple queries |
| Deployment (production) | Flask + AWS Bedrock | Managed memory, guardrails, multi-model access | AWS lock-in, mitigated by abstraction layer |

Every decision is constraint-driven. The architecture fits inside the constraints, not the other way around.