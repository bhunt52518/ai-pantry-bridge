# app/routers/recipes_parse.py
from __future__ import annotations

from fastapi import APIRouter

from app.clients.ollama import ollama  # matches your pattern used elsewhere
from app.models.recipe import NormalizedRecipe, RecipeParseRequest
from app.services.recipes_parse import parse_recipe_url

router = APIRouter(prefix="/recipe", tags=["recipe"])


@router.post("/parse", response_model=NormalizedRecipe)
async def recipe_parse(req: RecipeParseRequest) -> NormalizedRecipe:
    return await parse_recipe_url(url=req.url, ollama_client=ollama)
