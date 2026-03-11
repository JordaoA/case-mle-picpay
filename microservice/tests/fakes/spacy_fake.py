# tests/fakes/spacy_fake.py
from datetime import datetime, timezone

class FakeSpan:
    """Mimics spacy.tokens.Span"""
    def __init__(self, text: str, label_: str, start_char: int, end_char: int):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char

class FakeDoc:
    """Mimics spacy.tokens.Doc"""
    def __init__(self, text: str, ents: list[FakeSpan] = None):
        self.text = text
        self.ents = ents or []

class FakeLanguage:
    """Mimics spacy.language.Language (the 'nlp' object)"""
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name

    def __call__(self, text: str) -> FakeDoc:
        """
        When the router calls `doc = nlp(text)`, this method executes.
        We can add simple deterministic logic here to return entities 
        without needing a real ML model.
        """
        ents = []
        
        if "Apple" in text:
            start = text.find("Apple")
            ents.append(FakeSpan("Apple", "ORG", start, start + 5))
            
        if "New York" in text:
            start = text.find("New York")
            ents.append(FakeSpan("New York", "GPE", start, start + 8))
            
        return FakeDoc(text, ents)

class FakeModelManager:
    """Mimics your app.services.model_manager.ModelManager"""
    def get(self, model_name: str) -> FakeLanguage:
        return FakeLanguage(model_name)
        
    def ensure_available(self, model_name: str) -> str:
        return "downloaded"

class FakePredictionHistory:
    """An in-memory fake representing your Redis/Database persistence layer."""
    def __init__(self):
        self._store = []

    def save_prediction(self, input_text: str, model_name: str, entities: list) -> dict:
        record = {
            "id": f"pred_{len(self._store) + 1}",
            "input_text": input_text,
            "output": entities,
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._store.append(record)
        return record

    def get_all(self) -> list[dict]:
        return self._store
        
    def clear(self) -> None:
        self._store.clear()
        
class FakeMLflowRegistry:
    def __init__(self):
        self._registry = {}

    def register_model(self, model_name: str) -> None:
        self._registry[model_name] = {
            "name": model_name,
            "version": "1",
            "stage": "Production",
            "run_id": "fake-run-id"
        }

    def get_model_info(self, model_name: str) -> dict | None:
        return self._registry.get(model_name)

    def delete_registered_model(self, model_name: str) -> None:
        if model_name in self._registry:
            self._registry[model_name]["stage"] = "Archived"

    def list_registered_models(self) -> list[dict]:
        return list(self._registry.values())