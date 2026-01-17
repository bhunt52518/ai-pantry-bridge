# app/models/recipe.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecipeParseRequest(BaseModel):
    url: str


class NormalizedIngredient(BaseModel):
    raw: str
    item: str
    qty: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None


class NormalizedTime(BaseModel):
    prep_min: int = 0
    cook_min: int = 0
    total_min: int = 0


class NormalizedRecipe(BaseModel):
    title: str
    source_url: Optional[str] = None
    yield_: Optional[str] = Field(default=None, alias="yield")
    time: NormalizedTime = Field(default_factory=NormalizedTime)
    ingredients: List[NormalizedIngredient] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

class RecipeParseSaveResponse(BaseModel):
    recipe_id: int
    title: str
    source_url: Optional[str] = None


class RecipeListResponse(BaseModel):
    items: list[dict[str, Any]]


class RecipeGetResponse(BaseModel):
    recipe: dict[str, Any]
