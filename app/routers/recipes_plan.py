# app/routers/recipes_plan.py
from __future__ import annotations

from fastapi import APIRouter

from app.clients.ollama import ollama
from app.models.recipe import NormalizedRecipe
from app.models.recipe_plan import RecipePlanPayload, RecipePlanRequest
from app.services.recipes_parse import parse_recipe_url
from app.services.recipes_plan import build_speech, diff_recipe_against_pantry, post_to_callback

router = APIRouter(prefix="/recipe", tags=["recipe"])


@router.post("/plan", response_model=RecipePlanPayload)
async def recipe_plan(req: RecipePlanRequest) -> RecipePlanPayload:
    parsed: NormalizedRecipe = await parse_recipe_url(url=req.url, ollama_client=ollama)

    recipe_dict = parsed.model_dump(by_alias=True)
    missing, partial = diff_recipe_against_pantry(recipe_dict)

    speech = build_speech(parsed.title, missing, partial)

    payload = {
        "speech": speech,
        "url": req.url,
        "mode": req.mode,
        "recipe": {"title": parsed.title, "steps": parsed.steps},
        "missing": missing,
        "partial": partial,
    }

    # This is what triggers your HA automation -> phone notification
    await post_to_callback(req.callback_url, payload)

    return RecipePlanPayload(**payload)
