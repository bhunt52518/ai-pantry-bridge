AI Pantry & Recipe Bridge

Home Assistant–Integrated, Local-First Pantry & Recipe System

AI Pantry Bridge is a local-first pantry and recipe planning service designed to integrate tightly with Home Assistant.
It combines deterministic inventory logic with LLM-assisted understanding and language, while keeping all state changes predictable, auditable, and human-approved.

The system runs entirely on your local network using FastAPI, SQLite, and a local LLM (Qwen via Ollama).

Architecture Overview
Home Assistant
     |
     | REST / Webhooks
     v
AI Bridge (FastAPI) ────→ SQLite (Pantry, Recipes)
     |
     └──→ Ollama (Qwen)
           ^
           |
       Open WebUI

Design intent

Home Assistant orchestrates automations, notifications, and user interaction

AI Bridge owns all inventory and recipe logic

LLMs assist, but never mutate state directly

Components
Home Assistant

Automations, scripts, and mobile notifications

Voice (Siri / Assist / Echo)

Calls AI Bridge via REST

Receives results via webhooks

Does not manage inventory logic directly

AI Bridge (FastAPI)

Stateless HTTP API

Deterministic pantry + recipe logic

SQLite-backed persistence

Modular router/service architecture

Managed via systemd

Ollama / Qwen

Local LLM inference

Used only for:

Parsing recipe text / webpages

Generating natural language responses

Never:

Alters inventory

Makes truth decisions

Writes to the database

Open WebUI

Web UI for Ollama

Prompt iteration and testing

Not part of the runtime pipeline

Network & Ports (Defaults)
Component	Address / Port
Home Assistant	HOME_ASSISTANT_IP:8123
AI Bridge API	AI_BRIDGE_IP:8090
Open WebUI	AI_BRIDGE_IP:3000
Ollama API	127.0.0.1:11434

All addresses are configurable.

Database

SQLite (local file, not committed)

Tables

pantry – canonical inventory items

barcodes – barcode → canonical mappings

recipes – normalized parsed recipes

recipes_raw – raw recipe payload archive

Schema is created / validated at startup.

API Endpoints
Pantry

POST /pantry/upsert

POST /pantry/adjust

GET /pantry/items

GET /pantry/get/{name}

DELETE /pantry/delete/{name}

Barcodes

POST /barcode/teach

GET /barcode/resolve/{barcode}

Recipes

POST /recipe/parse
Parse a recipe URL into structured data (LLM-assisted)

POST /recipe/parse_and_save
Parse and persist a recipe

POST /recipe/plan
Parse recipe → diff against pantry → notify Home Assistant

GET /recipe/list

GET /recipe/get/{id}

Language / Speech

POST /speech/format
Structured data → natural language / SSML

Automation Generation (Optional)

POST /generate
LLM-assisted Home Assistant automation JSON generation
(Does not trigger pantry notifications)

Home Assistant Integration
Recipe planning → phone notification flow

HA calls POST /recipe/plan

AI Bridge:

Parses the recipe

Diffs ingredients vs pantry

Builds missing / partial

Sends payload to HA webhook

HA:

Stores payload

Sends actionable mobile notification

Buttons trigger follow-up automations

Example rest_command
rest_command:
  ai_bridge_recipe_plan:
    url: "http://AI_BRIDGE_IP:8090/recipe/plan"
    method: POST
    headers:
      Content-Type: application/json
    payload: >
      {
        "url": "{{ recipe_url }}",
        "callback_url": "http://HOME_ASSISTANT_IP:8123/api/webhook/AI_BRIDGE_WEBHOOK_ID"
      }

Design Principles
Deterministic Core

Inventory math

Canonical item naming

Explicit state changes only

LLM as Assistant

Parsing

Language generation

Suggestions, not decisions

Local-First

No cloud dependency

All data stored locally

Works offline

Human-in-the-Loop

Barcode teaching

Confirmation before shopping list changes

Actionable notifications

Safe Failure Modes

Invalid LLM output rejected

Inventory never mutates implicitly

Notifications are optional, not required

Status

✅ Pantry management
✅ Barcode learning
✅ Recipe parsing and storage
✅ Pantry diff + phone notifications
✅ Home Assistant integration
✅ Local LLM via Ollama

🚧 Ongoing:

Smarter quantity/unit reasoning

Voice-first flows

Multi-recipe planning

Setup Notes

Python 3.10+

FastAPI

SQLite

Ollama (local)

Home Assistant (recommended)

Exact setup steps vary by environment.
