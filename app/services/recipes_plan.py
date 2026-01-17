# app/services/recipes_plan.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import httpx

from app.services import common


def _canonicalize_ingredient_name(name: str) -> str:
    # Use your profiles-based canonicalization (same contract as pantry/barcode)
    profiles = common.load_profiles()
    return common.canonical_from_name(name, profiles)


def _get_pantry_set() -> Dict[str, float]:
    # canonical -> qty
    with common.pantry_db() as conn:
        rows = conn.execute("SELECT canonical, qty FROM pantry").fetchall()
    return {r[0]: float(r[1] or 0) for r in rows}


def diff_recipe_against_pantry(recipe: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Conservative diff:
      - missing: ingredient not in pantry OR qty <= 0
      - partial: ingredient in pantry but qty is >0 and <1
    (You can refine later once you add unit conversions / required qty parsing.)
    """
    pantry = _get_pantry_set()

    missing: List[str] = []
    partial: List[str] = []

    ingredients = recipe.get("ingredients") or []
    for ing in ingredients:
        # normalized recipe ingredient shape: {"raw","item","qty","unit","notes"}
        item = ""
        if isinstance(ing, dict):
            item = (ing.get("item") or ing.get("raw") or "").strip()
        elif isinstance(ing, str):
            item = ing.strip()

        if not item:
            continue

        canon = _canonicalize_ingredient_name(item)

        if canon not in pantry or pantry[canon] <= 0:
            missing.append(canon)
        elif 0 < pantry[canon] < 1:
            partial.append(canon)

    # unique, stable order
    def uniq(seq: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in seq:
            x = (x or "").strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return uniq(missing), uniq(partial)


def build_speech(title: str, missing: List[str], partial: List[str]) -> str:
    m = len(missing)
    p = len(partial)

    if m == 0 and p == 0:
        return f"You look good for {title}. I didn't find any missing items."

    parts = []
    if m:
        parts.append(f"{m} missing")
    if p:
        parts.append(f"{p} low")

    summary = " and ".join(parts)
    return f"For {title}, I found {summary}. Want to add them to your shopping list?"


async def post_to_callback(callback_url: str, payload: Dict[str, Any]) -> None:
    # Don’t swallow errors silently — if this fails, you won’t get a phone notification.
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.post(callback_url, json=payload)
        resp.raise_for_status()
