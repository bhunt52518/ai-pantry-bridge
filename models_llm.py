from pydantic import BaseModel, Field
from typing import List, Optional

class RecipeTime(BaseModel):
    prep_min: int = 0
    cook_min: int = 0
    total_min: int = 0

class Ingredient(BaseModel):
    raw: str
    item: str
    qty: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None

class NormalizedRecipe(BaseModel):
    title: str
    source_url: Optional[str] = None
    yield_: Optional[str] = Field(default=None, alias="yield")
    time: RecipeTime = Field(default_factory=RecipeTime)
    ingredients: List[Ingredient] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True  # allow yield_ <-> "yield"

class SpeechFormatResponse(BaseModel):
    speech: str
    display: Optional[str] = None
    ssml: Optional[str] = None

class RecipeParseRequest(BaseModel):
    url: str

class SpeechFormatRequest(BaseModel):
    text: str

class RecipeParseRequest(BaseModel):
    url: str

class RecipeTime(BaseModel):
    prep_min: int = 0
    cook_min: int = 0
    total_min: int = 0

class Ingredient(BaseModel):
    raw: str
    item: str
    qty: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None

class NormalizedRecipe(BaseModel):
    title: str
    source_url: Optional[str] = None
    yield_: Optional[str] = Field(default=None, alias="yield")
    time: RecipeTime = Field(default_factory=RecipeTime)
    ingredients: List[Ingredient] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True  # allow yield_ <-> "yield"
