from fastapi import FastAPI
from app.routers import speech, barcode, pantry, recipes_parse, recipes_ops, recipes_library, generate, recipes_plan

def create_app() -> FastAPI:
    app = FastAPI(title="AI Bridge")
    app.include_router(speech.router)
    app.include_router(barcode.router)
    app.include_router(pantry.router)
    app.include_router(recipes_parse.router)
    app.include_router(recipes_ops.router)
    app.include_router(recipes_library.router)
    app.include_router(generate.router)
    app.include_router(recipes_plan.router)
    return app

app = create_app()
