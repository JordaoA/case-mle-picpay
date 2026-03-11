from app.storage.history import _build_history # Your existing factory

_prediction_history = _build_history()

def get_history():
    """
    Dependency provider for the prediction history repository.
    This allows us to override the database in our tests.
    """
    return _prediction_history