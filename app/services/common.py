import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any
from app.core.config import PANTRY_DB, PROFILES_PATH


import httpx

def pantry_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(PANTRY_DB))
    conn.row_factory = sqlite3.Row
    return conn


def load_profiles() -> Dict[str, Any]:
    try:
        with open(str(PROFILES_PATH), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def canonical_from_name(name: str, profiles: Dict[str, Any]) -> str:
    n = normalize_name(name)

    if n in profiles:
        return n

    for canon, p in profiles.items():
        for a in p.get("aliases", []):
            if n == normalize_name(a):
                return canon

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
        (barcode.strip(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_barcode_mapping(barcode: str, label: str, canonical: str):
    conn = pantry_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO barcodes (barcode, label, canonical, updated_at) VALUES (?,?,?,?)",
        (barcode.strip(), label.strip(), canonical.strip(), now),
    )
    conn.commit()
    conn.close()


def upsert_pantry(canonical: str, add_qty: float, unit: str):
    conn = pantry_db()
    now = datetime.utcnow().isoformat()

    row = conn.execute(
        "SELECT qty, unit FROM pantry WHERE canonical=?",
        (canonical,),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO pantry (canonical, qty, unit, updated_at) VALUES (?,?,?,?)",
            (canonical, float(add_qty), unit, now),
        )
    else:
        existing_unit = row["unit"]
        final_unit = existing_unit if existing_unit != unit else unit
        conn.execute(
            "UPDATE pantry SET qty=?, unit=?, updated_at=? WHERE canonical=?",
            (float(row["qty"]) + float(add_qty), final_unit, now, canonical),
        )

    conn.commit()
    conn.close()


async def notify_unknown(callback_url: Optional[str], value: str):
    if not callback_url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(callback_url, json={"barcode": value})
    except Exception:
        pass
