import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from ollama_client import OllamaClient
from models_llm import SpeechFormatRequest, SpeechFormatResponse, RecipeParseRequest, NormalizedRecipe
import json
import re
import html as _html

# -------------------------
# Config
# -------------------------

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PANTRY_DB = os.path.join(DATA_DIR, "pantry.sqlite3")
PROFILES_PATH = os.path.join(DATA_DIR, "pantry_profiles.json")


SYSTEM_AUTOMATION = """You are a Home Assistant automation generator.

RULES:
- Output ONLY valid JSON.
- Do NOT include markdown, comments, or explanations.
- Do NOT include trailing commas.
- Do NOT include text outside the JSON object.
- All strings must be double-quoted.

The JSON schema MUST be:
{
  "valid": true | false,
  "summary": "short description",
  "assumptions": [],
  "helpers": [],
  "triggers": [],
  "conditions": [],
  "actions": [],
  "notes": []
}

REQUIREMENTS:
- Use Home Assistant service names exactly as documented.
- Entity IDs must be lowercase.
- If information is missing, set "valid" to false and explain in "notes".
- Prefer simplicity over cleverness.
- Never invent entities.
"""


app = FastAPI()

ollama = OllamaClient(base_url="http://127.0.0.1:11434", model="qwen2.5:14b")

# -------------------------
# Ollama Parser
# _________________________

def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output")
    return m.group(0)

def _title_tokens(s: str) -> set:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 4]
    return set(toks)

def _title_similar(a: str, b: str) -> bool:
    A = _title_tokens(a)
    B = _title_tokens(b)
    if not A or not B:
        return True  # don't block if we can't compare
    overlap = len(A & B)
    return overlap >= 2 or overlap >= max(1, min(len(A), len(B)) // 2)


@app.post("/speech/format", response_model=SpeechFormatResponse)
async def speech_format(req: SpeechFormatRequest):
    system = (
        "You format responses to be spoken by a smart home assistant.\n"
        "Return ONLY valid JSON. No markdown, no commentary.\n"
        "Schema:\n"
        '{ "speech": "string", "display": "string|null", "ssml": "string|null" }\n'
        "Rules:\n"
        "- speech must be 1–3 short sentences.\n"
        "- Avoid reading numbers or IDs unless necessary.\n"
        "- If there are options, end with a short question.\n"
    )
    user = f"Text to format:\n{req.text}"

    # attempt 1
    out = await ollama.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        timeout_s=60,
    )

    try:
        payload = json.loads(_extract_json(out))
        resp = SpeechFormatResponse(**payload)
        if not resp.speech.strip():
            raise ValueError("Empty speech")
        return resp

    except Exception:
        # retry once, stricter
        fix = "Your previous output was invalid. Return ONLY valid JSON matching the schema exactly."
        out2 = await ollama.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "user", "content": fix},
            ],
            temperature=0.0,
            timeout_s=60,
        )

        try:
            payload2 = json.loads(_extract_json(out2))
            resp2 = SpeechFormatResponse(**payload2)
            if not resp2.speech.strip():
                raise ValueError("Empty speech")
            return resp2
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM speech format failed: {e}")

def _find_jsonld_scripts(html: str) -> List[str]:
    """
    Extract contents of <script type="application/ld+json">...</script>.
    Regex-based (no bs4 dependency).
    """
    pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    blocks = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    # Unescape HTML entities just in case
    return [_html.unescape(b.strip()) for b in blocks if b and b.strip()]


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _extract_recipe_candidates_from_jsonld_text(jsonld_text: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON-LD script block and return any objects whose @type includes Recipe,
    including nested @graph structures.
    """
    candidates: List[Dict[str, Any]] = []

    try:
        data = json.loads(jsonld_text)
    except Exception:
        return candidates

    def walk(obj: Any):
        if isinstance(obj, dict):
            # If this dict is a Recipe
            types = obj.get("@type")
            type_list = [t.lower() for t in _as_list(types) if isinstance(t, str)]
            if "recipe" in type_list:
                candidates.append(obj)

            # Walk @graph if present
            if "@graph" in obj:
                for item in _as_list(obj.get("@graph")):
                    walk(item)

            # Walk all other values
            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return candidates


def _find_jsonld_scripts(html: str) -> List[str]:
    pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    blocks = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    return [_html.unescape(b.strip()) for b in blocks if b and b.strip()]

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _extract_recipe_candidates_from_jsonld_text(jsonld_text: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    try:
        data = json.loads(jsonld_text)
    except Exception:
        return candidates

    def walk(obj: Any):
        if isinstance(obj, dict):
            types = obj.get("@type")
            type_list = [t.lower() for t in _as_list(types) if isinstance(t, str)]
            if "recipe" in type_list:
                candidates.append(obj)

            if "@graph" in obj:
                for item in _as_list(obj.get("@graph")):
                    walk(item)

            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return candidates

def _jsonld_best_recipe(recipes: List[Dict[str, Any]], page_url: str) -> Optional[Dict[str, Any]]:
    if not recipes:
        return None

    page_url_norm = (page_url or "").strip().lower()

    def get_urlish(r: Dict[str, Any]) -> str:
        u = r.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip().lower()

        me = r.get("mainEntityOfPage")
        if isinstance(me, str) and me.strip():
            return me.strip().lower()
        if isinstance(me, dict):
            mid = me.get("@id") or me.get("url")
            if isinstance(mid, str) and mid.strip():
                return mid.strip().lower()
    matches = []
    for r in recipes:
        u = get_urlish(r)
        if u and page_url_norm and (page_url_norm in u or u in page_url_norm):
            matches.append(r)

    if not matches:
        return None

    # If multiple matches, prefer the one with the most ingredients
    def ing_len(r: Dict[str, Any]) -> int:
        ri = r.get("recipeIngredient")
        return len(ri) if isinstance(ri, list) else 0

    return sorted(matches, key=ing_len, reverse=True)[0]

    for r in recipes:
        u = get_urlish(r)
        if u and page_url_norm and (page_url_norm in u or u in page_url_norm):
            return r

    # Strong heuristic: prefer the one with the most ingredients if present
    def ing_len(r: Dict[str, Any]) -> int:
        ri = r.get("recipeIngredient")
        if isinstance(ri, list):
            return len(ri)
        return 0

    recipes_sorted = sorted(recipes, key=ing_len, reverse=True)
    return recipes_sorted[0]

def _extract_page_title(html: str) -> str:
    # Prefer og:title if present
    m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']', html, flags=re.I)
    if m:
        return m.group(1).strip()

    # Fallback to <title>
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    return ""


RECIPE_SCHEMA_HINT = """
Return ONLY valid JSON. No markdown, no commentary.

Schema:
{
  "title": "string",
  "source_url": "string|null",
  "yield": "string|null",
  "time": { "prep_min": 0, "cook_min": 0, "total_min": 0 },
  "ingredients": [
    {"raw":"string","item":"string","qty":"string|null","unit":"string|null","notes":"string|null"}
  ],
  "steps": ["string"],
  "tags": ["string"]
}

Rules:
- Always include "raw" for each ingredient.
- If a field is unknown, use null or an empty list.
- "item" should be the ingredient name (e.g., "onion", "soy sauce", "ground beef").
- steps should be short imperative sentences.
"""
@app.post("/recipe/parse", response_model=NormalizedRecipe)
async def recipe_parse(req: RecipeParseRequest):
    # 1) Fetch HTML
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(
                req.url,
                headers={"User-Agent": "AI-Bridge/1.0 (+HomeAssistant)"},
            )
            r.raise_for_status()
            html = r.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    # 2) Extract page title (ground truth)
    page_title = _extract_page_title(html)

    # 3) Extract JSON-LD recipe candidates
    jsonld_blocks = _find_jsonld_scripts(html)

    all_candidates: List[Dict[str, Any]] = []
    for block in jsonld_blocks:
        all_candidates.extend(_extract_recipe_candidates_from_jsonld_text(block))

    best_recipe = _jsonld_best_recipe(all_candidates, req.url)

    # 4) Build strict prompt rules
    common_rules = (
        f"Target URL (MUST MATCH): {req.url}\n"
        f"Page title (use as ground truth): {page_title}\n\n"
        "Hard rules:\n"
        "- Extract ONLY the main recipe from the page.\n"
        "- The recipe title must closely match the page title.\n"
        "- Do NOT substitute a different recipe.\n"
        "- Do NOT invent a recipe.\n"
        "- Return ONLY valid JSON matching the schema.\n"
    )

    if best_recipe:
        source_blob = json.dumps(best_recipe, ensure_ascii=False)
        system = (
            "You extract cooking recipes from structured JSON-LD into normalized JSON.\n"
            + RECIPE_SCHEMA_HINT
        )
        user = common_rules + "\nJSON-LD Recipe:\n" + source_blob
    else:
        html_snippet = html[:120000]
        system = (
            "You extract cooking recipes from webpage HTML into normalized JSON.\n"
            + RECIPE_SCHEMA_HINT
        )
        user = common_rules + "\nHTML:\n" + html_snippet

    # 5) Ask Qwen (deterministic)
    out = await ollama.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        timeout_s=180,
    )

    # 6) Validate JSON -> Pydantic
    try:
        payload = json.loads(_extract_json(out))
        payload["source_url"] = req.url
        parsed = NormalizedRecipe(**payload)

        # Title guard
        if page_title and parsed.title and not _title_similar(parsed.title, page_title):
            raise ValueError(f"title_mismatch: '{parsed.title}' vs '{page_title}'")

        return parsed

    except Exception:
        # 7) Retry once with stronger correction
        fix = (
            "Your previous output was invalid OR did not match the requested page.\n"
            f"The page title is: {page_title}\n"
            "Return ONLY the main recipe from THIS page.\n"
            "The recipe title MUST closely match the page title.\n"
            "Return ONLY valid JSON that matches the schema exactly.\n"
            "Do NOT invent a different recipe.\n"
        )

        out2 = await ollama.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "user", "content": fix},
            ],
            temperature=0.0,
            timeout_s=180,
        )

        try:
            payload2 = json.loads(_extract_json(out2))
            payload2["source_url"] = req.url
            parsed2 = NormalizedRecipe(**payload2)

            if page_title and parsed2.title and not _title_similar(parsed2.title, page_title):
                raise ValueError(f"title_mismatch: '{parsed2.title}' vs '{page_title}'")

            return parsed2

        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM parse failed: {e}")

class RecipeParseSaveResponse(BaseModel):
    recipe_id: str
    title: str
    source_url: Optional[str] = None

@app.post("/recipe/parse_and_save", response_model=RecipeParseSaveResponse)
async def recipe_parse_and_save(req: RecipeParseRequest):
    # 1) parse
    parsed = await recipe_parse(req)  # calls the endpoint function directly

    # 2) save (plug into your existing /recipe/save logic)
    # ---- BEGIN: replace this block with your real save logic ----
    # Example placeholder: create an ID and insert into sqlite
    recipe_id = f"r_{int(datetime.utcnow().timestamp())}"

    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(PANTRY_DB) as conn:
        cur = conn.cursor()
        # If you already have a recipes table, use it instead of this placeholder.
        # This is only an example. Use your existing schema/save code.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recipes_raw (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                source_url TEXT,
                json TEXT NOT NULL
            )
        """)
        cur.execute(
            "INSERT INTO recipes_raw (id, created_at, title, source_url, json) VALUES (?, ?, ?, ?, ?)",
            (recipe_id, datetime.utcnow().isoformat(), parsed.title, parsed.source_url, parsed.model_dump_json(by_alias=True)),
        )
        conn.commit()
    # ---- END: replace this block with your real save logic ----

    return RecipeParseSaveResponse(recipe_id=recipe_id, title=parsed.title, source_url=parsed.source_url)

# -------------------------
# DB helpers
# -------------------------

def pantry_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(PANTRY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = pantry_db()

    # Pantry core
    conn.execute("""
    CREATE TABLE IF NOT EXISTS pantry (
      canonical TEXT PRIMARY KEY,
      qty REAL NOT NULL,
      unit TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      category TEXT,
      perishable INTEGER NOT NULL DEFAULT 0,
      expires_at TEXT,
      staple INTEGER NOT NULL DEFAULT 0
    )
    """)

    # Barcode mappings
    conn.execute("""
    CREATE TABLE IF NOT EXISTS barcodes (
      barcode TEXT PRIMARY KEY,
      label TEXT NOT NULL,
      canonical TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
    """)

    # Recipes
    conn.execute("""
    CREATE TABLE IF NOT EXISTS recipes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      source_url TEXT,
      raw_text TEXT,
      parsed_json TEXT,
      updated_at TEXT NOT NULL
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recipes_title ON recipes(title)")

    conn.commit()
    conn.close()

@app.on_event("startup")
async def _startup():
    init_db()

def save_recipe(title: str, ingredients: list, source_url: Optional[str] = None, raw_text: Optional[str] = None):
    conn = pantry_db()
    now = datetime.utcnow().isoformat()
    parsed_json = json.dumps(ingredients, ensure_ascii=False)
    conn.execute(
        "INSERT INTO recipes (title, source_url, raw_text, parsed_json, updated_at) VALUES (?,?,?,?,?)",
        (title.strip(), source_url, raw_text, parsed_json, now)
    )
    conn.commit()
    conn.close()

def list_recipes():
    conn = pantry_db()
    rows = conn.execute(
        "SELECT id, title, source_url, updated_at FROM recipes ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_recipe(recipe_id: int):
    conn = pantry_db()
    row = conn.execute(
        "SELECT id, title, source_url, raw_text, parsed_json, updated_at FROM recipes WHERE id=?",
        (recipe_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None



# -------------------------
# Profile + normalization
# -------------------------

def load_profiles() -> Dict[str, Any]:
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def canonical_from_name(name: str, profiles: Dict[str, Any]) -> str:
    n = normalize_name(name)

    # direct match
    if n in profiles:
        return n

    # alias match
    for canon, p in profiles.items():
        for a in p.get("aliases", []):
            if n == normalize_name(a):
                return canon

    # fallback: use normalized string as canonical
    return n


def is_barcode(s: str) -> bool:
    s = s.strip()
    return s.isdigit() and 8 <= len(s) <= 16


def is_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def get_barcode_mapping(barcode: str) -> Optional[Dict[str, str]]:
    conn = pantry_db()
    row = conn.execute(
        "SELECT barcode, label, canonical FROM barcodes WHERE barcode=?",
        (barcode.strip(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_barcode_mapping(barcode: str, label: str, canonical: str):
    conn = pantry_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO barcodes (barcode, label, canonical, updated_at) VALUES (?,?,?,?)",
        (barcode.strip(), label.strip(), canonical.strip(), now)
    )
    conn.commit()
    conn.close()


def upsert_pantry(canonical: str, add_qty: float, unit: str):
    conn = pantry_db()
    now = datetime.utcnow().isoformat()

    row = conn.execute(
        "SELECT qty, unit FROM pantry WHERE canonical=?",
        (canonical,)
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO pantry (canonical, qty, unit, updated_at) VALUES (?,?,?,?)",
            (canonical, float(add_qty), unit, now)
        )
    else:
        # v1: if unit mismatches, keep existing unit (we’ll add conversions later)
        existing_unit = row["unit"]
        final_unit = existing_unit if existing_unit != unit else unit
        conn.execute(
            "UPDATE pantry SET qty=?, unit=?, updated_at=? WHERE canonical=?",
            (float(row["qty"]) + float(add_qty), final_unit, now, canonical)
        )

    conn.commit()
    conn.close()


async def notify_unknown(callback_url: Optional[str], value: str):
    """Notify Home Assistant that a barcode/QR needs labeling."""
    if not callback_url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(callback_url, json={"barcode": value})
    except Exception:
        pass


# -------------------------
# AI Bridge (/generate)
# -------------------------

class GenerateReq(BaseModel):
    model: str = "qwen2.5:14b"
    intent: str
    callback_url: str


def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


@app.post("/generate")
async def generate(req: GenerateReq):
    prompt = f"{SYSTEM_AUTOMATION}\n\nUSER INTENT:\n{req.intent}\n"

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(OLLAMA_URL, json={
            "model": req.model,
            "prompt": prompt,
            "stream": False
        })
        r.raise_for_status()
        raw = r.json().get("response", "").strip()

    try:
        parsed = safe_json_parse(raw)
        summary = parsed.get("summary", "Draft created")
        out = {
            "summary": summary,
            "raw": json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        }
    except Exception as e:
        out = {
            "summary": "Invalid JSON from model",
            "raw": json.dumps({
                "valid": False,
                "summary": "Invalid JSON from model",
                "assumptions": [],
                "helpers": [],
                "triggers": [],
                "conditions": [],
                "actions": [],
                "notes": [f"{type(e).__name__}: {e}"]
            })
        }

    # callback to HA
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(req.callback_url, json=out)
    except Exception:
        pass

    return {"ok": True}


# -------------------------
# Pantry endpoints
# -------------------------

class PantryAddReq(BaseModel):
    name: str
    qty: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    callback_url: Optional[str] = None  # HA webhook for "unknown barcode"
    category: Optional[str] = None
    perishable: Optional[bool] = None
    expires_at: Optional[str] = None  # ISO string
    staple: Optional[bool] = None

@app.post("/pantry/add")
async def pantry_add(req: PantryAddReq):
    profiles = load_profiles()
    incoming = req.name.strip()

    # 1) QR codes / URLs: treat like barcodes (look up mapping first)
    if is_url(incoming):
        mapping = get_barcode_mapping(incoming)
        if not mapping:
            await notify_unknown(req.callback_url, incoming)
            return {"ok": True, "added": False, "unknown_barcode": True, "value": incoming}
        canon = mapping["canonical"]

    # 2) Numeric barcodes (UPC/EAN): resolve via mapping table
    elif is_barcode(incoming):
        mapping = get_barcode_mapping(incoming)
        if not mapping:
            await notify_unknown(req.callback_url, incoming)
            return {"ok": True, "added": False, "unknown_barcode": True, "barcode": incoming}
        canon = mapping["canonical"]

    # 3) Normal named items
    else:
        canon = canonical_from_name(incoming, profiles)

    # Determine qty/unit
    add_qty = 1.0
    unit = "count"

    p = profiles.get(canon)

    if req.qty is not None:
        add_qty = float(req.qty)
        unit = (req.unit or "count").strip().lower()
    elif p is not None:
        add_qty = float(p.get("default_add_qty", 1.0))
        unit = str(p.get("unit", "count")).strip().lower()

    # add means "increase qty", so fetch current first
    existing = get_pantry_item(canon)
    current_qty = float(existing["qty"]) if existing else 0.0
    new_qty = current_qty + float(add_qty)
    set_pantry_qty(
        canon,
        new_qty,
        unit,
        category=req.category,
        perishable=req.perishable,
        expires_at=req.expires_at,
        staple=req.staple
    )


    return {
        "ok": True,
        "added": True,
        "canonical": canon,
        "added_amount": {"qty": add_qty, "unit": unit}
    }


@app.get("/pantry/list")
async def pantry_list():
    conn = pantry_db()
    rows = conn.execute(
        "SELECT canonical, qty, unit, category, perishable, expires_at, staple, updated_at "
        "FROM pantry ORDER BY canonical"
    ).fetchall()
    conn.close()
    return {"ok": True, "items": [dict(r) for r in rows]}


class PantryDeleteReq(BaseModel):
    canonical: str


@app.post("/pantry/delete")
async def pantry_delete(req: PantryDeleteReq):
    canon = normalize_name(req.canonical)
    conn = pantry_db()
    cur = conn.execute("DELETE FROM pantry WHERE canonical=?", (canon,))
    conn.commit()
    conn.close()
    return {"ok": True, "deleted": cur.rowcount, "canonical": canon}


def get_pantry_item(canonical: str):
    conn = pantry_db()
    row = conn.execute(
        "SELECT canonical, qty, unit, category, perishable, expires_at, staple, updated_at "
        "FROM pantry WHERE canonical=?",
        (canonical,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def set_pantry_qty(
    canonical: str,
    qty: float,
    unit: str,
    *,
    category: Optional[str] = None,
    perishable: Optional[bool] = None,
    expires_at: Optional[str] = None,
    staple: Optional[bool] = None
):
    conn = pantry_db()
    now = datetime.utcnow().isoformat()

    # Fetch existing so we don't wipe metadata on REPLACE
    existing = conn.execute(
        "SELECT category, perishable, expires_at, staple FROM pantry WHERE canonical=?",
        (canonical,)
    ).fetchone()

    existing_category = existing["category"] if existing else None
    existing_perishable = int(existing["perishable"]) if existing else 0
    existing_expires_at = existing["expires_at"] if existing else None
    existing_staple = int(existing["staple"]) if existing else 0

    final_category = (category.strip().lower() if isinstance(category, str) and category.strip()
                      else existing_category)
    final_perishable = (1 if perishable else 0) if perishable is not None else existing_perishable
    final_expires_at = expires_at if expires_at is not None else existing_expires_at
    final_staple = (1 if staple else 0) if staple is not None else existing_staple

    if qty <= 0:
        conn.execute("DELETE FROM pantry WHERE canonical=?", (canonical,))
    else:
        conn.execute(
            "INSERT OR REPLACE INTO pantry "
            "(canonical, qty, unit, updated_at, category, perishable, expires_at, staple) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (canonical, qty, unit, now, final_category, final_perishable, final_expires_at, final_staple)
        )

    conn.commit()
    conn.close()


class PantryConsumeReq(BaseModel):
    canonical: str
    qty: float
    unit: Optional[str] = None  # optional, v1 will require matching unit

@app.post("/pantry/consume")
async def pantry_consume(req: PantryConsumeReq):
    canon = normalize_name(req.canonical)
    item = get_pantry_item(canon)
    if not item:
        return {"ok": False, "error": "not_found", "canonical": canon}

    current_qty = float(item["qty"])
    current_unit = item["unit"]

    use_unit = (req.unit or current_unit).strip().lower()
    if use_unit != current_unit:
        # v1: no conversions yet (we can add later)
        return {
            "ok": False,
            "error": "unit_mismatch",
            "canonical": canon,
            "current_unit": current_unit,
            "requested_unit": use_unit
        }

    new_qty = max(0.0, current_qty - float(req.qty))
    set_pantry_qty(canon, new_qty, current_unit)

    return {
        "ok": True,
        "canonical": canon,
        "from": {"qty": current_qty, "unit": current_unit},
        "to": {"qty": new_qty, "unit": current_unit}
    }

class PantrySetReq(BaseModel):
    canonical: str
    qty: float
    unit: Optional[str] = None  # if omitted, keeps existing unit (or defaults to count if new)
    category: Optional[str] = None
    perishable: Optional[bool] = None
    expires_at: Optional[str] = None
    staple: Optional[bool] = None

@app.post("/pantry/set")
async def pantry_set(req: PantrySetReq):
    canon = normalize_name(req.canonical)

    # If item exists, keep its unit unless caller explicitly sets unit
    existing = get_pantry_item(canon)
    if existing:
        current_unit = existing["unit"]
        target_unit = (req.unit or current_unit).strip().lower()
        if target_unit != current_unit and req.unit is not None:
            # v1: no conversions (we can add later)
            return {
                "ok": False,
                "error": "unit_mismatch",
                "canonical": canon,
                "current_unit": current_unit,
                "requested_unit": target_unit
            }
        final_unit = current_unit if req.unit is None else target_unit
    else:
        # New item: default unit is count unless provided
        final_unit = (req.unit or "count").strip().lower()

    new_qty = max(0.0, float(req.qty))
    set_pantry_qty(
        canon,
        new_qty,
        final_unit,
        category=req.category,
        perishable=req.perishable,
        expires_at=req.expires_at,
        staple=req.staple
    )

    return {
        "ok": True,
        "canonical": canon,
        "to": {"qty": new_qty, "unit": final_unit}
    }


# -------------------------
# Barcode endpoints
# -------------------------

class BarcodeTeachReq(BaseModel):
    barcode: str
    label: str


@app.post("/barcode/teach")
async def barcode_teach(req: BarcodeTeachReq):
    profiles = load_profiles()
    canon = canonical_from_name(req.label, profiles)
    set_barcode_mapping(req.barcode.strip(), req.label.strip(), canon)
    return {"ok": True, "barcode": req.barcode.strip(), "label": req.label.strip(), "canonical": canon}


# NOTE: This endpoint works best for numeric barcodes.
# URLs contain slashes and may not work nicely in a path parameter.
# Use /barcode/teach for URLs (it stores fine), and verify via DB or scans.
@app.get("/barcode/resolve/{barcode}")
async def barcode_resolve(barcode: str):
    m = get_barcode_mapping(barcode.strip())
    if not m:
        return {"ok": False, "known": False, "barcode": barcode.strip()}
    return {"ok": True, "known": True, "barcode": m["barcode"], "label": m["label"], "canonical": m["canonical"]}


# -----------------------
# Recipe endpoints
# -----------------------

class RecipeIng(BaseModel):
    canonical: str
    qty: Optional[float] = 1.0
    unit: Optional[str] = "count"
    raw: Optional[str] = None

class RecipeDiffReq(BaseModel):
    ingredients: List[RecipeIng]
    mode: Literal["strict", "lenient"] = "strict"

class RecipeApplyReq(BaseModel):
    ingredients: List[RecipeIng]               # reuse RecipeIng from diff
    mode: Literal["strict", "lenient"] = "strict"
    dry_run: bool = False
    consume_staples: bool = False 

class RecipeSaveReq(BaseModel):
    title: str
    source_url: Optional[str] = None
    ingredients: List[RecipeIng]   # reuse your RecipeIng from diff
    raw_text: Optional[str] = None

class RecipePlanReq(BaseModel):
    max_results: int = 10
    include_partials: bool = True

def get_pantry_map():
    """
    Returns dict:
      canon -> {
        canonical, qty, unit, category, perishable, expires_at, staple, updated_at
      }
    """
    conn = pantry_db()
    rows = conn.execute(
        "SELECT canonical, qty, unit, category, perishable, expires_at, staple, updated_at FROM pantry"
    ).fetchall()
    conn.close()
    return {r["canonical"]: dict(r) for r in rows}


@app.post("/recipe/diff")
async def recipe_diff(req: RecipeDiffReq):
    pantry = get_pantry_map()

    used = []
    missing = []
    partial = []
    notes = []

    def norm_unit(u: Optional[str]) -> str:
        return (u or "count").strip().lower()

    def norm_qty(q: Optional[float]) -> float:
        try:
            return float(q if q is not None else 1.0)
        except Exception:
            return 1.0

    for ing in req.ingredients:
        canon = normalize_name(ing.canonical)
        need_qty = max(0.0, norm_qty(ing.qty))
        need_unit = norm_unit(ing.unit)

        # Ignore zero-qty lines
        if need_qty <= 0:
            continue

        item = pantry.get(canon)

        # Not in pantry
        if not item:
            missing.append({
                "canonical": canon,
                "need": need_qty,
                "unit": need_unit,
                "have": 0.0,
                "raw": ing.raw
            })
            continue

        have_qty = float(item.get("qty") or 0.0)
        have_unit = norm_unit(item.get("unit"))
        is_staple = int(item.get("staple") or 0) == 1

        # Staple rule: assume you have it (unless qty is explicitly 0)
        if is_staple and have_qty > 0:
            used.append({
                "canonical": canon,
                "need": need_qty,
                "unit": need_unit,
                "have": have_qty,
                "have_unit": have_unit,
                "staple": True,
                "raw": ing.raw
            })
            continue

        # Unit mismatch handling
        if need_unit != have_unit:
            notes.append({
                "type": "unit_mismatch",
                "canonical": canon,
                "need": {"qty": need_qty, "unit": need_unit},
                "have": {"qty": have_qty, "unit": have_unit},
                "raw": ing.raw
            })

            # strict: treat as missing (we're not converting yet)
            if req.mode == "strict":
                missing.append({
                    "canonical": canon,
                    "need": need_qty,
                    "unit": need_unit,
                    "have": have_qty,
                    "have_unit": have_unit,
                    "raw": ing.raw
                })
                continue

            # lenient (very conservative): if either is count, still treat as missing
            missing.append({
                "canonical": canon,
                "need": need_qty,
                "unit": need_unit,
                "have": have_qty,
                "have_unit": have_unit,
                "raw": ing.raw
            })
            continue

        # Quantity compare
        if have_qty >= need_qty:
            used.append({
                "canonical": canon,
                "need": need_qty,
                "unit": need_unit,
                "have": have_qty,
                "raw": ing.raw
            })
        elif have_qty > 0:
            partial.append({
                "canonical": canon,
                "need": need_qty,
                "unit": need_unit,
                "have": have_qty,
                "missing_qty": max(0.0, need_qty - have_qty),
                "raw": ing.raw
            })
        else:
            missing.append({
                "canonical": canon,
                "need": need_qty,
                "unit": need_unit,
                "have": 0.0,
                "raw": ing.raw
            })

    can_make = (len(missing) == 0 and len(partial) == 0)

    return {
        "ok": True,
        "can_make": can_make,
        "used": used,
        "partial": partial,
        "missing": missing,
        "notes": notes
    }

@app.post("/recipe/apply")
async def recipe_apply(req: RecipeApplyReq):
    # First: compute what would happen
    diff = await recipe_diff(RecipeDiffReq(ingredients=req.ingredients, mode=req.mode))

    if not diff.get("ok"):
        return diff

    # If anything is missing/partial, refuse to apply (safe default)
    if diff["missing"] or diff["partial"]:
        return {
            "ok": False,
            "error": "insufficient_inventory",
            "can_make": False,
            "diff": diff
        }

    # Dry-run returns the diff + what would be consumed
    if req.dry_run:
        return {
            "ok": True,
            "applied": False,
            "dry_run": True,
            "diff": diff
        }

    # Apply: consume used items, skipping staples unless consume_staples=true
    applied = []
    skipped = []
    errors = []

    for u in diff["used"]:
        canon = normalize_name(u["canonical"])
        need_qty = float(u.get("need", 0.0))
        need_unit = (u.get("unit") or "count").strip().lower()

        # Re-check current pantry state (avoid stale diff)
        item = get_pantry_item(canon)
        if not item:
            errors.append({"canonical": canon, "error": "not_found_at_apply"})
            continue

        is_staple = int(item.get("staple") or 0) == 1
        if is_staple and not req.consume_staples:
            skipped.append({"canonical": canon, "reason": "staple_not_consumed"})
            continue

        current_qty = float(item["qty"])
        current_unit = (item["unit"] or "count").strip().lower()

        if current_unit != need_unit:
            errors.append({
                "canonical": canon,
                "error": "unit_mismatch",
                "current_unit": current_unit,
                "requested_unit": need_unit
            })
            continue

        if current_qty < need_qty:
            errors.append({
                "canonical": canon,
                "error": "insufficient_at_apply",
                "have": current_qty,
                "need": need_qty,
                "unit": current_unit
            })
            continue

        new_qty = max(0.0, current_qty - need_qty)
        set_pantry_qty(canon, new_qty, current_unit)
        applied.append({
            "canonical": canon,
            "from": {"qty": current_qty, "unit": current_unit},
            "to": {"qty": new_qty, "unit": current_unit},
            "consumed": {"qty": need_qty, "unit": current_unit}
        })

    ok = (len(errors) == 0)

    return {
        "ok": ok,
        "applied": ok,
        "diff": diff,
        "applied_items": applied,
        "skipped": skipped,
        "errors": errors
    }

@app.post("/recipe/save")
async def recipe_save(req: RecipeSaveReq):
    ing_dicts = [i.model_dump() for i in req.ingredients]
    save_recipe(req.title, ing_dicts, source_url=req.source_url, raw_text=req.raw_text)
    return {"ok": True, "saved": True, "title": req.title}

@app.get("/recipe/list")
async def recipe_list():
    return {"ok": True, "recipes": list_recipes()}

@app.get("/recipe/get/{recipe_id}")
async def recipe_get(recipe_id: int):
    r = get_recipe(recipe_id)
    if not r:
        return {"ok": False, "error": "not_found", "id": recipe_id}
    r["ingredients"] = json.loads(r["parsed_json"] or "[]")
    return {"ok": True, "recipe": r}
@app.post("/recipe/plan")
async def recipe_plan(req: RecipePlanReq):
    conn = pantry_db()
    rows = conn.execute("SELECT id, title, source_url, parsed_json FROM recipes").fetchall()
    conn.close()

    pantry = get_pantry_map()
    ranked = []

    def perish_weight(canon: str) -> float:
        item = pantry.get(canon)
        if not item:
            return 0.0
        w = 0.0
        if int(item.get("perishable") or 0) == 1:
            w += 1.0
        if item.get("expires_at"):
            w += 1.0
        return w

    for r in rows:
        ingredients = json.loads(r["parsed_json"] or "[]")

        diff = await recipe_diff(RecipeDiffReq(
            ingredients=[RecipeIng(**i) for i in ingredients],
            mode="strict"
        ))

        missing_count = len(diff["missing"])
        partial_count = len(diff["partial"])

        if not req.include_partials and (missing_count > 0 or partial_count > 0):
            continue

        used_perish = sum(perish_weight(u["canonical"]) for u in diff["used"])
        score = (missing_count * 1000) + (partial_count * 100) - used_perish

        ranked.append({
            "id": r["id"],
            "title": r["title"],
            "source_url": r["source_url"],
            "can_make": (missing_count == 0 and partial_count == 0),
            "missing_count": missing_count,
            "partial_count": partial_count,
            "missing": diff["missing"],
            "partial": diff["partial"],
            "score": score
        })

    ranked.sort(key=lambda x: x["score"])
    return {"ok": True, "ranked": ranked[: max(1, req.max_results)]}

class RecipeParseAndSaveResponse(BaseModel):
    ok: bool = True
    saved: bool = True
    title: str = ""
    source_url: Optional[str] = None


@app.post("/recipe/parse_and_save", response_model=RecipeParseAndSaveResponse)
async def recipe_parse_and_save(req: RecipeParseRequest):
    # 1) Parse the recipe (re-use the in-process function, not HTTP)
    parsed: NormalizedRecipe = await recipe_parse(req)

    # 2) Convert NormalizedRecipe -> RecipeSaveReq shape
    # Your /recipe/save expects:
    # - title
    # - ingredients: list of Pydantic ingredient objects (then model_dump())
    # - source_url
    # - raw_text
    #
    # We’ll store raw_text as a compact text version (steps + ingredients raw).
    raw_text_parts: List[str] = []
    raw_text_parts.append(parsed.title or "")
    if parsed.ingredients:
        raw_text_parts.append("\nINGREDIENTS:")
        raw_text_parts.extend([f"- {i.raw}" for i in parsed.ingredients if i.raw])
    if parsed.steps:
        raw_text_parts.append("\nSTEPS:")
        raw_text_parts.extend([f"{idx+1}. {s}" for idx, s in enumerate(parsed.steps) if s])
    raw_text = "\n".join([p for p in raw_text_parts if p])

    # Build ingredients as dicts to match your save_recipe usage
    ing_dicts = []
    for i in parsed.ingredients:
        ing_dicts.append({
            # keep the fields you already use in your DB
            # if your schema expects other keys, add them here
            "name": i.item,                # common internal field name
            "item": i.item,                # keep both if you’re unsure; remove one if you know
            "quantity": i.qty,
            "qty": i.qty,
            "unit": i.unit,
            "notes": i.notes,
            "raw": i.raw,
        })

    # 3) Save using the exact same function your /recipe/save uses
    save_recipe(
        parsed.title,
        ing_dicts,
        source_url=parsed.source_url or req.url,
        raw_text=raw_text,
    )

    return RecipeParseAndSaveResponse(
        ok=True,
        saved=True,
        title=parsed.title,
        source_url=parsed.source_url or req.url,
    )

class RecipeParseDiffReq(BaseModel):
    url: str
    mode: Literal["strict", "lenient"] = "strict"

def _parse_qty_to_float(q: Optional[str]) -> float:
    """
    Best-effort parser:
    - "1" -> 1.0
    - "1.5" -> 1.5
    - "1/2" -> 0.5
    - "1 1/2" -> 1.5
    - None/"" -> 1.0
    """
    if not q:
        return 1.0
    s = str(q).strip()

    # handle "1 1/2"
    if " " in s:
        parts = s.split()
        try:
            whole = float(parts[0])
            frac = parts[1]
            if "/" in frac:
                num, den = frac.split("/", 1)
                return whole + (float(num) / float(den))
        except Exception:
            pass

    # handle "1/2"
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        except Exception:
            return 1.0

    # normal float
    try:
        return float(s)
    except Exception:
        return 1.0

@app.post("/recipe/parse_diff")
async def recipe_parse_diff(req: RecipeParseDiffReq):
    # 1) Parse recipe from URL
    parsed: NormalizedRecipe = await recipe_parse(RecipeParseRequest(url=req.url))

    # 2) Convert parsed ingredients -> RecipeDiffReq ingredients
    diff_ings = []
    for ing in parsed.ingredients:
        canonical = (ing.item or "").strip().lower()
        if not canonical:
            # fallback to raw if item is missing
            canonical = (ing.raw or "").strip().lower()

        diff_ings.append(
            RecipeIng(
                canonical=canonical,
                qty=_parse_qty_to_float(ing.qty),
                unit=(ing.unit or "count").strip().lower(),
                raw=ing.raw or f"{ing.qty or ''} {ing.unit or ''} {ing.item or ''}".strip(),
            )
        )

    # 3) Run your existing diff
    diff_req = RecipeDiffReq(ingredients=diff_ings, mode=req.mode)
    result = await recipe_diff(diff_req)

    # 4) Attach recipe context (handy for HA)
    result["recipe"] = {
        "title": parsed.title,
        "source_url": parsed.source_url or req.url
    }
    return result
@app.post("/recipe/parse_diff_speech")
async def recipe_parse_diff_speech(req: RecipeParseDiffReq):
    # 1) Get diff
    diff = await recipe_parse_diff(req)

    # 2) Create a compact summary for the formatter
    missing = diff.get("missing", []) or []
    partial = diff.get("partial", []) or []
    can_make = bool(diff.get("can_make"))

    title = (diff.get("recipe") or {}).get("title") or "this recipe"

    if can_make:
        summary = f"You can make {title} with what you have."
        if diff.get("used"):
            summary += f" It will use {min(len(diff.get('used')), 5)} pantry items."
    else:
        need_count = len(missing) + len(partial)
        top_missing = []
        for m in (missing[:3] + partial[:3]):
            canon = m.get("canonical")
            if canon and canon not in top_missing:
                top_missing.append(canon)
            if len(top_missing) >= 3:
                break
        if top_missing:
            summary = f"You can’t fully make {title}. You’re short {need_count} items, like {', '.join(top_missing)}."
        else:
            summary = f"You can’t fully make {title}. You’re short {need_count} items."

    # 3) Format for speech via your existing /speech/format logic (call function directly)
    speech_resp = await speech_format(SpeechFormatRequest(text=summary))

    return {
        "ok": True,
        "diff": diff,
        "speech": speech_resp.speech,
        "display": speech_resp.display,
        "ssml": speech_resp.ssml,
    }

HA_WEBHOOK_SPEECH_URL = "http://192.168.68.72:8123/api/webhook/ai_bridge_speech"

@app.post("/recipe/parse_diff_notify")
async def recipe_parse_diff_notify(req: RecipeParseDiffReq):
    # 1) Use your existing parse+diff endpoint
    diff = await recipe_parse_diff(req)

    # 2) Make a compact summary for speech formatting
    missing = diff.get("missing", []) or []
    partial = diff.get("partial", []) or []
    can_make = bool(diff.get("can_make"))
    title = (diff.get("recipe") or {}).get("title") or "that recipe"

    if can_make:
        summary = f"You can make {title} with what you have. Want me to show the steps?"
    else:
        need_count = len(missing) + len(partial)
        top = []
        for m in (missing + partial):
            c = m.get("canonical")
            if c and c not in top:
                top.append(c)
            if len(top) >= 3:
                break
        if top:
            summary = f"You can’t fully make {title}. You’re short {need_count} items, like {', '.join(top)}. Want me to add them to your shopping list?"
        else:
            summary = f"You can’t fully make {title}. You’re short {need_count} items. Want me to add them to your shopping list?"

    # 3) Format speech using your existing formatter
    speech_obj = await speech_format(SpeechFormatRequest(text=summary))

    # 4) Push to HA webhook (HA automation sends phone notification)
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(HA_WEBHOOK_SPEECH_URL, json={"speech": speech_obj.speech})
        r.raise_for_status()

    return {"ok": True, "notified": True, "speech": speech_obj.speech}
