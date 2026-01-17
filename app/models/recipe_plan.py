# app/models/recipe_plan.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RecipePlanRequest(BaseModel):
    url: str
    callback_url: str  # this should be your Nabu Casa cloudhook or /api/webhook/<id>
    mode: str = "recipe_plan"


class RecipePlanPayload(BaseModel):
    speech: str
    url: str
    mode: str
    recipe: Dict[str, Any]
    missing: List[str]
    partial: List[str]
