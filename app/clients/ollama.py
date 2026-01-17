from ollama_client import OllamaClient

# Keep these the same as your working config
ollama = OllamaClient(
    base_url="http://127.0.0.1:11434",
    model="qwen2.5:14b",
)
