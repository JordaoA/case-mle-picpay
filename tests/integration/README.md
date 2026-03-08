# Integration Tests

This directory contains integration tests for the **API routers** in the microservice, testing real HTTP endpoints with mocked external dependencies.

## Structure

```
tests/integration/
├── __init__.py                      # Package marker
├── conftest.py                      # Shared fixtures for integration tests
├── test_models_router.py            # Tests for models endpoints
└── test_predictions_router.py       # Tests for predictions endpoints
```

## Test Coverage

### Models Router (`test_models_router.py`)

Tests for model lifecycle management endpoints:

| Endpoint | Method | Tests |
|----------|--------|-------|
| `/load/` | POST | Download & register, already available, errors |
| `/models/` | GET | List all, empty list, response format |
| `/models/{model_name}` | DELETE | Success, not found, special chars, errors |

**Key Test Classes:**
- `TestLoadModelEndpoint` - POST /load/ endpoint
- `TestListModelsEndpoint` - GET /models/ endpoint  
- `TestDeleteModelEndpoint` - DELETE /models/{model_name} endpoint
- `TestModelsRouterIntegration` - Full workflows

### Predictions Router (`test_predictions_router.py`)

Tests for NER inference and history endpoints:

| Endpoint | Method | Tests |
|----------|--------|-------|
| `/predict/` | POST | Success, empty entities, errors, edge cases |
| `/list/` | GET | Success, empty, multiple records, format |

**Key Test Classes:**
- `TestPredictEndpoint` - POST /predict/ endpoint
- `TestListPredictionsEndpoint` - GET /list/ endpoint
- `TestPredictionsRouterIntegration` - Workflows
- `TestPredictionsResponseFormat` - Response validation

## Running Tests

### Run all integration tests:
```bash
pytest tests/integration/ -v
```

### Run specific test file:
```bash
pytest tests/integration/test_models_router.py -v
```

### Run specific test class:
```bash
pytest tests/integration/test_models_router.py::TestLoadModelEndpoint -v
```

### Run specific test:
```bash
pytest tests/integration/test_models_router.py::TestLoadModelEndpoint::test_load_model_success_downloaded -v
```

### Run with coverage:
```bash
pytest tests/integration/ --cov=app --cov-report=term-missing
```

## Test Features

### Mocking Strategy
- Uses `@patch` decorators to mock external services
- Mocks are scoped at the router module level
- Maintains isolation from database, Redis, and MLflow

### Fixtures
- `client` - FastAPI TestClient for making HTTP requests
- `sample_entities` - Pre-built entity examples
- `sample_entities` - Pre-built entity examples
- `mock_prediction_history` - Mocked history storage
- `sample_predict_response` - Real response structure
- `sample_model_info` - Real model info structure

### Test Types

**Endpoint Tests:**
- Happy path (success cases)
- Error handling (400, 404, 500 errors)
- Input validation (missing/invalid fields)
- Edge cases (empty results, special characters, unicode)

**Workflow Tests:**
- Load → List → Delete flow
- Predict → List history flow
- Multiple consecutive operations
- Different models/inputs

**Format Validation:**
- Response schema compliance
- Required field presence
- Data type correctness
- Timestamp formatting

## Dependencies

Integration tests require:
- `pytest`
- `pytest-mock`
- `fastapi`
- `httpx` (via TestClient)

These are in `tests/requirements-test.txt`

## Notes

- Tests use `FastAPI.TestClient` for in-memory testing (no actual HTTP)
- External services (MLflow, Redis, model loading) are mocked
- Tests are independent and can run in any order
- Use `pytest -k` flag to filter by test name patterns

## Adding More Tests

When adding new endpoints:

1. Create test class: `TestMyEndpoint`
2. Add fixtures as needed in `conftest.py`
3. Use `@patch` for service mocking
4. Follow naming: `test_<scenario>_<expected_behavior>`
5. Test both success and failure paths
