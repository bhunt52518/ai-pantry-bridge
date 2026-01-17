from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.recipe import RecipeGetResponse, RecipeListResponse
from app.services import recipes_repo

router = APIRouter(prefix="/recipe", tags=["recipe"])


@router.get("/list", response_model=RecipeListResponse)
def recipe_list(limit: int = 50) -> RecipeListResponse:
    items = recipes_repo.list_recipes(limit=limit)
    return RecipeListResponse(items=items)


@router.get("/get/{recipe_id}", response_model=RecipeGetResponse)
def recipe_get(recipe_id: int) -> RecipeGetResponse:
    r = recipes_repo.get_recipe(recipe_id)
    if not r:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return RecipeGetResponse(recipe=r)
