# app/services/recipes_parse.py
from __future__ import annotations

import html as _html
import json
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from app.core.text import extract_json
from app.models.recipe import NormalizedRecipe

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


def _title_tokens(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 4]
    return set(toks)


def _title_similar(a: str, b: str) -> bool:
    A = _title_tokens(a)
    B = _title_tokens(b)
    if not A or not B:
        return True
    overlap = len(A & B)
    return overlap >= 2 or overlap >= max(1, min(len(A), len(B)) // 2)


def _extract_page_title(html: str) -> str:
    m = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        html,
        flags=re.I,
    )
    if m:
        return m.group(1).strip()

    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    return ""


def _find_jsonld_scripts(html: str) -> List[str]:
    pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    blocks = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    return [_html.unescape(b.strip()) for b in blocks if b and b.strip()]


def _as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _extract_recipe_candidates_from_jsonld_text(jsonld_text: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    try:
        data = json.loads(jsonld_text)
    except Exception:
        return candidates

    def walk(obj: Any) -> None:
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

        return ""

    matches: List[Dict[str, Any]] = []
    for r in recipes:
        u = get_urlish(r)
        if u and page_url_norm and (page_url_norm in u or u in page_url_norm):
            matches.append(r)

    # If we can’t match by URL, fall back to “most ingredients”
    pool = matches if matches else recipes

    def ing_len(r: Dict[str, Any]) -> int:
        ri = r.get("recipeIngredient")
        return len(ri) if isinstance(ri, list) else 0

    return sorted(pool, key=ing_len, reverse=True)[0]


async def parse_recipe_url(*, url: str, ollama_client: Any) -> NormalizedRecipe:
    # 1) Fetch HTML
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "AI-Bridge/1.0 (+HomeAssistant)"})
            r.raise_for_status()
            html = r.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    # 2) Page title ground truth
    page_title = _extract_page_title(html)

    # 3) JSON-LD candidates
    jsonld_blocks = _find_jsonld_scripts(html)
    candidates: List[Dict[str, Any]] = []
    for block in jsonld_blocks:
        candidates.extend(_extract_recipe_candidates_from_jsonld_text(block))

    best_recipe = _jsonld_best_recipe(candidates, url)

    # 4) Build strict prompt rules
    common_rules = (
        f"Target URL (MUST MATCH): {url}\n"
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
        system = "You extract cooking recipes from structured JSON-LD into normalized JSON.\n" + RECIPE_SCHEMA_HINT
        user = common_rules + "\nJSON-LD Recipe:\n" + source_blob
    else:
        html_snippet = html[:120000]
        system = "You extract cooking recipes from webpage HTML into normalized JSON.\n" + RECIPE_SCHEMA_HINT
        user = common_rules + "\nHTML:\n" + html_snippet

    async def _call(messages: list[dict[str, str]]) -> NormalizedRecipe:
        out = await ollama_client.chat(messages, temperature=0.0, timeout_s=180)
        payload = json.loads(extract_json(out))
        payload["source_url"] = url
        parsed = NormalizedRecipe(**payload)

        if page_title and parsed.title and not _title_similar(parsed.title, page_title):
            raise ValueError(f"title_mismatch: '{parsed.title}' vs '{page_title}'")

        return parsed

    # 5) Ask model (attempt 1)
    try:
        return await _call([{"role": "system", "content": system}, {"role": "user", "content": user}])
    except Exception:
        # 6) Retry once, stronger correction
        fix = (
            "Your previous output was invalid OR did not match the requested page.\n"
            f"The page title is: {page_title}\n"
            "Return ONLY the main recipe from THIS page.\n"
            "The recipe title MUST closely match the page title.\n"
            "Return ONLY valid JSON that matches the schema exactly.\n"
            "Do NOT invent a different recipe.\n"
        )
        try:
            return await _call(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "user", "content": fix},
                ]
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM parse failed: {e}")
