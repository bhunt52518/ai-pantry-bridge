# app/routers/recipes_ops.py
from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException

from app.clients.ollama import ollama
from app.models.recipe import NormalizedRecipe, RecipeParseRequest, RecipeParseSaveResponse
from app.services.recipes_parse import parse_recipe_url
from app.services import recipes_repo

router = APIRouter(prefix="/recipe", tags=["recipe"])


@router.post("/parse_and_save", response_model=RecipeParseSaveResponse)
async def parse_and_save(req: RecipeParseRequest) -> RecipeParseSaveResponse:
    parsed: NormalizedRecipe = await parse_recipe_url(url=req.url, ollama_client=ollama)

    # Save normalized recipe into `recipes`
    rid = recipes_repo.save_recipe_parsed(
        title=parsed.title,
        source_url=parsed.source_url,
        parsed=parsed.model_dump(by_alias=True),
        raw_text=None,
    )

    # Optional: also archive into recipes_raw (nice for debugging/regeneration)
    # Use stable-ish id string so it never collides with int ids.
    raw_id = f"r_{rid}"
    try:
        recipes_repo.save_recipe_raw(
            recipe_id=raw_id,
            title=parsed.title,
            source_url=parsed.source_url,
            payload_json=json.dumps(parsed.model_dump(by_alias=True), ensure_ascii=False),
        )
    except Exception:
        # Don't fail the request if raw archive insert fails
        pass

    return RecipeParseSaveResponse(recipe_id=rid, title=parsed.title, source_url=parsed.source_url)
