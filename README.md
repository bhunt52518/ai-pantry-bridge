# ai-pantry-bridge-README.md

# AI Pantry & Recipe Planner (Home Assistant Integration)

A local-first pantry and recipe planning system integrated with Home Assistant, backed by SQLite and augmented by a local LLM (Qwen via Ollama).

The system combines **deterministic inventory logic** with **LLM-powered understanding and natural language**, while keeping all state changes safe and predictable.

---

## Architecture Overview

Home Assistant
|
| REST
v
AI Bridge (FastAPI)
|
| SQLite
v
Pantry Database

AI Bridge ──→ Ollama (Qwen)
^
|
Open WebUI


---

## Components

### Home Assistant
- Handles automations, scripts, notifications, and voice
- Calls the AI Bridge via REST
- Does not manage inventory logic directly

### AI Bridge (FastAPI)
- Stateless HTTP API
- Deterministic inventory and recipe logic
- SQLite-backed persistence
- Systemd-managed service

### Ollama / Qwen
- Local LLM inference
- Used **only** for:
  - Parsing human recipe text into structured data
  - Generating natural-language responses
- Never mutates inventory or makes truth decisions

### Open WebUI
- Web UI for interacting with Ollama models
- Used for prompt development and testing

---

## Network & Ports (Configurable)

| Component      | Default |
|----------------|---------|
| Home Assistant | `HOME_ASSISTANT_IP:8123` |
| AI Bridge API  | `AI_BRIDGE_IP:8090` |
| Open WebUI     | `AI_BRIDGE_IP:3000` |
| Ollama API     | `127.0.0.1:11434` |

> All addresses are configurable. Defaults assume a local network.

---

## Database

**SQLite file**


### Tables
- `pantry`
- `barcodes`
- `recipes`

Database schema is automatically created/validated at startup.

---

## API Endpoints

### Pantry
- `POST /pantry/add`
- `GET  /pantry/list`
- `POST /pantry/set`
- `POST /pantry/consume`
- `POST /pantry/delete`

### Barcodes
- `POST /barcode/teach`
- `GET  /barcode/resolve/{barcode}`

### Recipes
- `POST /recipe/save`
- `GET  /recipe/list`
- `GET  /recipe/get/{id}`
- `POST /recipe/diff`
- `POST /recipe/plan`
- `POST /recipe/apply`

### LLM-Backed (Optional)
- `POST /recipe/parse` – recipe text → structured ingredients
- `POST /speech/format` – structured data → natural language

---

## Home Assistant Integration

Example `rest_command`:

```yaml
rest_command:
  ai_bridge_recipe_plan:
    url: "http://AI_BRIDGE_IP:8090/recipe/plan"
    method: POST
    headers:
      Content-Type: application/json
    payload: >
      {"max_results": {{ max_results | default(5) }}}

A sample script is included to answer:

“What can I make?”

Design Principles

Deterministic Core

Inventory math

Unit handling

Consumption rules

LLM as Assistant

Text parsing

Language generation

Local-first

No cloud dependency

All data stored locally

Human-in-the-loop

Barcode teaching

Alias confirmation

Safe failure modes

Status

✅ Pantry management complete

✅ Recipe planning and ranking

✅ Home Assistant integration

✅ Local LLM support via Ollama

🚧 Ongoing: advanced parsing and voice interaction

Setup Notes

Python 3.10+

FastAPI

SQLite

Ollama (local)

Home Assistant (optional but recommended)

Exact setup steps may vary depending on environment.
