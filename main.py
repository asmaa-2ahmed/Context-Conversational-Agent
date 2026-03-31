import gradio as gr
from fastapi import FastAPI

from api.routers import health, chat, tools
from ui.gradio_app import demo

app = FastAPI(title="Conversational Agent")

# ── Register routers ──────────────────────────────────
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(tools.router)

# ── Mount Gradio ──────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")


