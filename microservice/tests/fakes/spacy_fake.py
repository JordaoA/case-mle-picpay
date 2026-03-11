"""
Fake implementations of spaCy components and the ModelManager.
Used to isolate testing from heavy ML model loading.
"""

class FakeSpan:
    """Mimics spacy.tokens.Span."""

    def __init__(self, text: str, label_: str, start_char: int, end_char: int):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class FakeDoc:
    """Mimics spacy.tokens.Doc."""

    def __init__(self, text: str, ents: list[FakeSpan] | None = None):
        self.text = text
        self.ents = ents or []


class FakeLanguage:
    """Mimics spacy.language.Language (the 'nlp' object)."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name

    def __call__(self, text: str) -> FakeDoc:
        """
        Returns hardcoded entities for specific keywords to ensure
        deterministic test assertions.
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
    """Mimics app.services.model_manager.ModelManager."""

    def __init__(self) -> None:
        self._loaded: dict[str, FakeLanguage] = {"en_core_web_sm": FakeLanguage()}

    def get(self, model_name: str) -> FakeLanguage:
        if model_name not in self._loaded:
            raise ValueError(f"Model '{model_name}' is not available.")
        return self._loaded[model_name]

    def ensure_available(self, model_name: str) -> str:
        self._loaded[model_name] = FakeLanguage(model_name)
        return "downloaded"

    def delete(self, model_name: str) -> None:
        self._loaded.pop(model_name, None)

    def list_models(self) -> list[str]:
        return list(self._loaded.keys())