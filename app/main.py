from fastapi import FastAPI
from app.routers import speech, barcode, pantry, recipes_parse, recipes_ops, recipes_library, generate, recipes_plan
from app.routers import health
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware


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
    app.include_router(health.router)

    setup_logging()

    app.add_middleware(RequestLoggingMiddleware)

    return app

app = create_app()
