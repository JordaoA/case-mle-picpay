# Unit Tests

This directory contains unit tests for the **core services and utilities** in the microservice, testing individual components in isolation with mocked dependencies.

## Structure

```
tests/unit/
├── __init__.py                              # Package marker
├── test_config.py                           # Tests for configuration settings
├── services/
│   ├── test_mlflow_registry.py              # Tests for MLflow registry service
│   ├── test_model_manager.py                # Tests for model manager service
│   └── test_ner_service.py                  # Tests for NER inference service
└── storage/
    ├── test_history_factory.py              # Tests for history factory
    ├── test_history_inmemory.py             # Tests for in-memory history
    └── test_redis_history.py                # Tests for Redis history
```

## Test Coverage

### Configuration (`test_config.py`)

Tests for the `Settings` configuration class that manages environment variables:

| Component | Tests |
|-----------|-------|
| MLflow settings | host, port, experiment name |
| Service settings | name, version |
| Redis settings | host, port, db, password, TTL |
| Tracking URI | construction and formatting |

**Key Test Class:**
- `TestSettings` - Settings initialization and properties

### Services

#### MLflow Registry (`services/test_mlflow_registry.py`)

Tests for MLflow model registry integration:

| Functionality | Tests |
|---------------|-------|
| Registry initialization | MLflow client setup |
| Model registration | Create, retrieve, list models |
| Model versioning | Create versions, transition stages |
| Run tracking | Create, log parameters and metrics |

**Key Test Class:**
- `TestMLflowRegistry` - MLflow registry operations

#### Model Manager (`services/test_model_manager.py`)

Tests for model lifecycle management:

| Functionality | Tests |
|---------------|-------|
| Model availability | Check installation, download |
| Model loading | Load from cache, spaCy integration |
| Model listing | List available and loaded models |
| Model deletion | Remove from cache |

**Key Test Class:**
- `TestModelManager` - Model management operations

#### NER Service (`services/test_ner_service.py`)

Tests for Named Entity Recognition inference:

| Functionality | Tests |
|---------------|-------|
| Prediction | Run NER on text, extract entities |
| Model loading | Get model from manager |
| Response formatting | Build prediction response |
| Error handling | Invalid models, missing text |

**Key Test Class:**
- `TestNERService` - NER inference operations

### Storage

#### History Factory (`storage/test_history_factory.py`)

Tests for the history storage factory pattern:

| Functionality | Tests |
|---------------|-------|
| Factory creation | Select correct history backend |
| Backend initialization | In-memory vs Redis selection |
| Configuration override | Switch between implementations |

**Key Test Class:**
- `TestHistoryFactory` - History storage factory

#### In-Memory History (`storage/test_history_inmemory.py`)

Tests for in-memory prediction history implementation:

| Functionality | Tests |
|---------------|-------|
| Add records | Single and multiple predictions |
| Retrieve records | Get all, by index, by ID |
| Counter management | ID auto-increment |
| Data integrity | Records preservation |

**Key Test Class:**
- `TestInMemoryHistory` - In-memory storage operations

#### Redis History (`storage/test_redis_history.py`)

Tests for Redis-backed prediction history:

| Functionality | Tests |
|---------------|-------|
| Connection | Redis client initialization |
| Add records | Persist to Redis |
| Retrieve records | Fetch from Redis |
| TTL management | Expiration handling |
| Pipeline operations | Batch operations |

**Key Test Class:**
- `TestRedisHistory` - Redis storage operations

## Running Tests

### Run all unit tests:
```bash
pytest tests/unit/ -v
```

### Run specific test file:
```bash
pytest tests/unit/test_config.py -v
pytest tests/unit/services/test_model_manager.py -v
pytest tests/unit/storage/test_history_inmemory.py -v
```

### Run specific test class:
```bash
pytest tests/unit/services/test_ner_service.py::TestNERService -v
```

### Run specific test:
```bash
pytest tests/unit/test_config.py::TestSettings::test_settings_mlflow_defaults -v
```

### Run only unit-marked tests:
```bash
pytest tests/unit/ -v -m unit
```

### Run with coverage:
```bash
pytest tests/unit/ --cov=app --cov-report=term-missing
```

## Test Features

### Mocking Strategy
- Uses `@patch` decorators to mock external dependencies
- Mocks external services: MLflow, Redis, spaCy, subprocess
- Maintains isolation from actual system resources
- Fixtures provide reusable mock objects

### Fixtures
- `env_vars` - Environment variables for configuration
- `mock_settings` - Mocked Settings object
- `mock_mlflow` - Mocked MLflow module
- `mock_client` - Mocked MLflow client
- `mock_spacy` - Mocked spaCy module
- `mock_model_manager` - Mocked ModelManager
- `mock_prediction_history` - Mocked history storage
- `mock_redis_client` - Mocked Redis client
- `sample_entity_result` - Sample entity for predictions
- `sample_prediction_record` - Sample prediction record

### Test Types

**Configuration Tests:**
- Environment variable loading
- Default value handling
- Type validation
- URI construction

**Service Tests:**
- Happy path (success cases)
- Error conditions (exceptions, invalid inputs)
- Edge cases (empty results, special characters)
- State management

**Storage Tests:**
- CRUD operations (Create, Read, Update, Delete)
- Data persistence
- Expiration/TTL management
- Concurrent access

**Integration Between Components:**
- Service-to-storage interactions
- Configuration to service initialization
- Error propagation

## Dependencies

Unit tests require:
- `pytest>=7.0.0`
- `pytest-mock>=3.10.0`
- `pytest-cov>=4.0.0`
- `requests-mock>=1.9.0`
- `fakeredis>=2.19.0` (for Redis mocking)

These are in `tests/requirements-test.txt`

## Mark Patterns

Tests use pytest markers for organization:

```bash
# Run only unit tests
pytest -m unit

# Run without specific markers
pytest -m "not integration"
```

## Best Practices

### Writing New Unit Tests

1. **Isolate dependencies**: Mock all external services
2. **Test behavior**: Focus on what the function does, not implementation
3. **Clear naming**: Test name should describe the scenario and expected outcome
4. **One assertion per test**: Keep tests focused and granular
5. **Use fixtures**: Leverage reusable mock objects from conftest

### Test Organization

- One test file per module being tested
- Group related tests in classes (e.g., `TestModelManager`)
- Use setup fixtures for common mock objects
- Document complex test scenarios with docstrings

### Debugging Tests

Enable verbose output:
```bash
pytest tests/unit/ -vv --tb=short
```

Show print statements:
```bash
pytest tests/unit/ -s
```

Stop on first failure:
```bash
pytest tests/unit/ -x
```

## Notes

- Unit tests run in isolation without network or file system access
- All external dependencies are mocked for fast, reliable execution
- Tests follow the Arrange-Act-Assert (AAA) pattern
- Fixtures are defined in `conftest.py` for sharing across tests
- Use `@pytest.mark.unit` to explicitly mark unit tests
- Tests should be independent and can run in any order

## Adding More Unit Tests

When testing new services:

1. Create test file: `test_<service_name>.py`
2. Create test class: `Test<ServiceName>`
3. Add fixtures in `conftest.py` if reusable
4. Follow naming: `test_<scenario>_<expected_behavior>`
5. Mock all external dependencies
6. Test success and failure paths
