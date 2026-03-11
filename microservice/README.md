# Part 2 — NER Microservice: Running Guide

This guide covers everything you need to run the NER Inference Service using the provided `Makefile`.

All commands assume you are inside the `microservice/` directory:

```bash
cd case-mle-picpay/microservice
```

---

## Prerequisites

- Docker installed and running
- `make` available on your system (pre-installed on macOS and Linux)
- No Python installation required to run the service

---

## Quick start

```bash
make up
```

That's it. This single command will:

1. Build the NER service Docker image (multi-stage, ~300 MB — takes ~3–5 min on first run)
2. Pull `mongo:6.0` and the MLflow image
3. Start all three containers in the correct order, waiting for each health check to pass
4. Print the service URLs when everything is ready

Expected output when healthy:

```
── Building and starting full stack ──
── Waiting for services to be healthy ──
   Waiting for picpay-mongo …
   ✔ picpay-mongo is healthy.
   Waiting for picpay-mlflow …
   ✔ picpay-mlflow is healthy.
   Waiting for picpay-ner-service …
   ✔ picpay-ner-service is healthy.

✔ Stack is up and healthy.

   API        → http://localhost:8000
   Swagger UI → http://localhost:8000/docs
   MLflow UI  → http://localhost:5000

   Run make logs to tail output.
   Run make down to stop.
```

---

## All available commands

| Command | Description |
|---|---|
| `make up` | Build images and start the full stack |
| `make down` | Stop all containers (data volumes are preserved) |
| `make clean` | Stop containers **and** delete all volumes (MLflow + Redis data) |
| `make logs` | Tail live logs from all containers (`Ctrl+C` to stop) |
| `make test` | Run the unit test suite on the host |
| `make install` | Install Python dependencies on the host (needed for `make test`) |
| `make help` | Print the command reference |

---

## Configuring the environment

You can create a `.env` file to override defaults. Simply set any of the variables listed below:

```bash
cp .env.example .env  # If .env.example exists
# Or create .env manually with variables below
```

Key variables (optional — defaults shown):

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_HOST` | `mlflow` | MLflow hostname |
| `MLFLOW_PORT` | `5000` | MLflow port |
| `MLFLOW_EXPERIMENT_NAME` | `ner-inference-service` | MLflow experiment name |
| `MONGO_URI` | `mongodb://mongo:27017` | MongoDB connection URI |
| `MONGO_DB_NAME` | `picpay_ner` | MongoDB database name |
| `SERVICE_NAME` | `picpay-ner-service` | Service identifier used in MLflow tags |
| `SERVICE_VERSION` | `1.0.0` | Service version |
| `SERVICE_PORT` | `8000` | Services HTTP port |

> Do not use quotes around values and do not add inline comments in `.env`.

---

## Typical usage after `make up`

### 1. Load a model

```bash
curl -X POST http://localhost:8000/load/ \
  -H "Content-Type: application/json" \
  -d '{"model": "en_core_web_sm"}'
```

Available models: `en_core_web_sm` (~12 MB), `en_core_web_md` (~43 MB), `en_core_web_lg` (~741 MB).

### 2. Run a prediction

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple acquired Beats for $3 billion in 2014.", "model": "en_core_web_sm"}'
```

### 3. Check prediction history

```bash
curl http://localhost:8000/list/
```

### 4. Inspect the model registry

```bash
curl http://localhost:8000/models/
# or open the MLflow UI:
open http://localhost:5000
```

### 5. Check service health

```bash
curl http://localhost:8000/health/
```

### 6. Explore interactively

Open **http://localhost:8000/docs** for the Swagger UI, which lets you call every endpoint directly from the browser.

---

## Stopping the service

```bash
# Stop containers — all MLflow and Redis data is preserved in Docker volumes
make down

# Stop containers AND wipe all data (useful for a clean restart)
make clean
```

After `make down`, a subsequent `make up` will restart in ~10 seconds without rebuilding images.

---

## Running the tests

The unit tests run entirely on the host using `fakeredis` — no running containers needed.

```bash
# Install dependencies (only needed once)
make install

# Run the test suite
make test
```

To pass extra pytest flags:

```bash
make test ARGS="-v --tb=long"
make test ARGS="tests/unit/services/test_model_manager.py"
```

---

## Troubleshooting

**`make up` times out waiting for `picpay-ner-service`**

Check the container logs for the actual error:

```bash
docker compose logs ner-service --tail=50
```

Common causes: missing `.env` variable, MLflow not yet reachable on startup, or a Python exception in the app.

**Port already in use**

Stop whatever is using port 8000 or 5000 first, or edit the port mappings in `docker-compose.yaml`.

**Starting fresh after a bad state**

```bash
make clean   # removes containers + volumes
make up      # full rebuild from scratch
```
