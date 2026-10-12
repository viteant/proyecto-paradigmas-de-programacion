"""Cliente HTTP aislado para el servidor local de Ollama."""

import requests


class OllamaError(RuntimeError):
    """Error de conexión o de formato al consultar Ollama."""


class OllamaClient:
    """Consulta el modelo local sin mezclar HTTP con análisis estadístico."""

    def __init__(self, model: str = "llama3.2:3b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def status(self) -> tuple[bool, list[str]]:
        """Comprueba servicio y lista los modelos instalados."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            models = [
                item.get("name", "")
                for item in response.json().get("models", [])
            ]
            return True, models
        except (requests.RequestException, ValueError):
            return False, []

    def chat(self, messages: list[dict], json_mode: bool = False,
             max_tokens: int = 450) -> str:
        """Envía una conversación acotada al modelo configurado.

        Con ``json_mode`` Ollama restringe la salida a un objeto JSON.
        """
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0, "seed": 42, "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=180
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as error:
            raise OllamaError(str(error)) from error
