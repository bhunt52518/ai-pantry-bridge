# app/routers/pantry.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from app.services import common

router = APIRouter(prefix="/pantry", tags=["pantry"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: str) -> str:
    profiles = common.load_profiles()
    return common.canonical_from_name(value, profiles)


@router.get("/health")
def health() -> dict[str, Any]:
    with common.pantry_db() as conn:
        conn.execute("SELECT 1").fetchone()
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pantry'").fetchone()
    return {"ok": True}


@router.get("/items")
def list_items() -> dict[str, Any]:
    with common.pantry_db() as conn:
        rows = conn.execute(
            """
            SELECT
              canonical, qty, unit, updated_at,
              category, perishable, expires_at, staple
            FROM pantry
            ORDER BY canonical COLLATE NOCASE
            """
        ).fetchall()

    items = []
    for r in rows:
        canonical, qty, unit, updated_at, category, perishable, expires_at, staple = r
        items.append(
            {
                "canonical": canonical,
                "qty": qty,
                "unit": unit,
                "updated_at": updated_at,
                "category": category,
                "perishable": bool(perishable),
                "expires_at": expires_at,
                "staple": bool(staple),
            }
        )

    return {"items": items}


@router.get("/get/{name_or_canonical}")
def get_item(name_or_canonical: str) -> dict[str, Any]:
    canonical = _canonical(name_or_canonical)

    with common.pantry_db() as conn:
        row = conn.execute(
            """
            SELECT
              canonical, qty, unit, updated_at,
              category, perishable, expires_at, staple
            FROM pantry
            WHERE canonical = ?
            """,
            (canonical,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Item not found")

    canonical, qty, unit, updated_at, category, perishable, expires_at, staple = row
    return {
        "item": {
            "canonical": canonical,
            "qty": qty,
            "unit": unit,
            "updated_at": updated_at,
            "category": category,
            "perishable": bool(perishable),
            "expires_at": expires_at,
            "staple": bool(staple),
        }
    }


@router.post("/upsert")
def upsert(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Upsert / set quantity (absolute).
    Example payload:
      {
        "name": "Diced Tomatoes",
        "qty": 4,
        "unit": "can",
        "category": "canned",
        "perishable": false,
        "expires_at": "2026-12-01",
        "staple": true
      }
    """
    raw = payload.get("name") or payload.get("canonical")
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="Missing 'name' (or 'canonical')")

    qty = payload.get("qty", 1)
    if not isinstance(qty, (int, float)):
        raise HTTPException(status_code=400, detail="Invalid 'qty'")

    canonical = _canonical(raw)

    unit = payload.get("unit") or ""  # unit is NOT NULL in schema
    category = payload.get("category")
    perishable = 1 if bool(payload.get("perishable", 0)) else 0
    expires_at = payload.get("expires_at")
    staple = 1 if bool(payload.get("staple", 0)) else 0

    updated_at = _now_iso()

    with common.pantry_db() as conn:
        conn.execute(
            """
            INSERT INTO pantry
              (canonical, qty, unit, updated_at, category, perishable, expires_at, staple)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical) DO UPDATE SET
              qty = excluded.qty,
              unit = excluded.unit,
              updated_at = excluded.updated_at,
              category = excluded.category,
              perishable = excluded.perishable,
              expires_at = excluded.expires_at,
              staple = excluded.staple
            """,
            (canonical, float(qty), str(unit), updated_at, category, perishable, expires_at, staple),
        )
        conn.commit()

    return {"ok": True, "canonical": canonical, "qty": float(qty), "unit": str(unit)}


@router.post("/adjust")
def adjust(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Adjust quantity by delta (positive or negative).
    If qty becomes <= 0, deletes the row.
    Example:
      {"name":"Diced Tomatoes","delta":-1}
    """
    raw = payload.get("name") or payload.get("canonical")
    delta = payload.get("delta")

    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="Missing 'name' (or 'canonical')")
    if delta is None or not isinstance(delta, (int, float)):
        raise HTTPException(status_code=400, detail="Missing/invalid 'delta'")

    canonical = _canonical(raw)
    updated_at = _now_iso()

    with common.pantry_db() as conn:
        row = conn.execute(
            "SELECT qty, unit FROM pantry WHERE canonical = ?",
            (canonical,),
        ).fetchone()

        # If missing and delta > 0, create it with defaults
        if not row:
            if float(delta) <= 0:
                raise HTTPException(status_code=404, detail="Item not found")
            conn.execute(
                """
                INSERT INTO pantry
                  (canonical, qty, unit, updated_at, category, perishable, expires_at, staple)
                VALUES
                  (?, ?, ?, ?, NULL, 0, NULL, 0)
                """,
                (canonical, float(delta), "", updated_at),
            )
            conn.commit()
            return {"ok": True, "created": True, "canonical": canonical, "qty": float(delta), "unit": ""}

        current_qty, unit = row
        new_qty = float(current_qty) + float(delta)

        if new_qty <= 0:
            conn.execute("DELETE FROM pantry WHERE canonical = ?", (canonical,))
            conn.commit()
            return {"ok": True, "deleted": True, "canonical": canonical, "qty": 0, "unit": unit}

        conn.execute(
            "UPDATE pantry SET qty = ?, updated_at = ? WHERE canonical = ?",
            (new_qty, updated_at, canonical),
        )
        conn.commit()

    return {"ok": True, "canonical": canonical, "qty": new_qty, "unit": unit}


@router.delete("/delete/{name_or_canonical}")
def delete_item(name_or_canonical: str) -> dict[str, Any]:
    canonical = _canonical(name_or_canonical)

    with common.pantry_db() as conn:
        cur = conn.execute("DELETE FROM pantry WHERE canonical = ?", (canonical,))
        conn.commit()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Item not found")

    return {"ok": True, "deleted": True, "canonical": canonical}
