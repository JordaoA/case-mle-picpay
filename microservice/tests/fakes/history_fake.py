"""
Fake implementation of the MongoDB prediction history repository.
"""

from datetime import datetime

from app.schemas.requests import EntityResult, PredictionRecord


class FakePredictionHistory:
    """An in-memory fake representing the MongoDB persistence layer."""

    def __init__(self) -> None:
        self._store: list[PredictionRecord] = []
        self._is_healthy: bool = True

    def add(
        self,
        input_text: str,
        output: list[EntityResult],
        model: str,
        timestamp: datetime,
    ) -> PredictionRecord:
        """Saves a prediction and assigns a fake MongoDB ObjectId."""
        record = PredictionRecord(
            id=f"65a2b1c3d4e5f6g7h8i9j0k{len(self._store)}",
            input_text=input_text,
            output=output,
            model=model,
            timestamp=timestamp,
        )
        self._store.append(record)
        return record

    def all(self) -> list[PredictionRecord]:
        """Returns all records, newest first, mimicking MongoDB sort."""
        return list(reversed(self._store))

    def count(self) -> int:
        return len(self._store)

    def ping(self) -> bool:
        return self._is_healthy

    def clear(self) -> None:
        self._store.clear()