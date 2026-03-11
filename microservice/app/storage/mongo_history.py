# microservice/app/storage/mongo_history.py
import logging
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from app.config import settings
from app.schemas.requests import EntityResult, PredictionRecord

logger = logging.getLogger("ner_service.mongo_history")

class MongoHistory:
    """Thread-safe prediction history backed by MongoDB."""
    def __init__(self, client=None) -> None:
        if client:
            self._client = client
        else:
            self._client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
            
        self._db = self._client[settings.mongo_db_name]
        self._collection = self._db["predictions"]
        self._counters = self._db["counters"]
        
        logger.info(f"MongoHistory initialised — DB: {settings.mongo_db_name}")

    def _get_next_id(self) -> int:
        """Atomic counter to mimic Redis INCR and maintain integer IDs."""
        counter_doc = self._counters.find_one_and_update(
            {"_id": "prediction_id"},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=True
        )
        return counter_doc["seq"]

    def add(
        self,
        input_text: str,
        output: list[EntityResult],
        model: str,
        timestamp: datetime,
    ) -> PredictionRecord:
        
        prediction_id = str(self._get_next_id())
        
        record = PredictionRecord(
            id=prediction_id,
            input_text=input_text,
            output=output,
            model=model,
            timestamp=timestamp,
        )

        document = {
            "prediction_id": prediction_id,
            "input_text": input_text,
            "output": [e.model_dump() for e in output],
            "model": model,
            "timestamp": timestamp,
        }
        
        self._collection.insert_one(document)
        logger.debug(f"Stored prediction id={prediction_id} in MongoDB")
        return record

    def all(self) -> list[PredictionRecord]:
        cursor = self._collection.find().sort("timestamp", -1)
        
        records = []
        for doc in cursor:
            try:
                records.append(
                    PredictionRecord(
                        id=doc["prediction_id"],
                        input_text=doc["input_text"],
                        output=[EntityResult(**e) for e in doc["output"]],
                        model=doc["model"],
                        timestamp=doc["timestamp"],
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to deserialise prediction record: {exc}")
                
        return records

    def count(self) -> int:
        return self._collection.count_documents({})

    def ping(self) -> bool:
        try:
            self._client.admin.command('ismaster')
            return True
        except ConnectionFailure:
            return False