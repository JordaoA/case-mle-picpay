# Part 2 — NER Microservice: Running Guide

This guide covers everything you need to run the NER Inference Service using the provided `Makefile`.

All commands assume you are inside the `microservice/` directory:

```bash
cd case-mle-picpay/microservice

```

---

## Prerequisites

* Docker installed and running
* `make` available on your system
* Python 3.x (required for running tests on the host)

---

## Quick start

```bash
make up

```

This command builds the NER service image and starts the full stack (FastAPI, MongoDB, and MLflow), waiting for all health checks to pass before finishing.

Expected output when healthy:

```
✔ Stack is up and healthy.

   API        → http://localhost:8000
   Swagger UI → http://localhost:8000/docs
   MLflow UI  → http://localhost:5000

```

---

## All available commands

| Command | Description |
| --- | --- |
| `make up` | Build images and start the full stack |
| `make down` | Stop containers, preserving volumes |
| `make clean` | Stop containers and delete all volumes (wipes MLflow/Mongo data) |
| `make logs` | Tail live logs from all containers |
| `make test` | Run the full test suite (unit & integration) |
| `make test-unit` | Run only unit tests |
| `make test-int` | Run only integration tests |
| `make install` | Install all app and test dependencies on the host |

---

## Running the tests

To run the test cases, you must set up a local Python environment to provide the necessary dependencies for `pytest`.

1. **Create and activate a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


2. **Install dependencies**:
```bash
make install

```


This command upgrades `pip` and installs requirements from both `requirements.txt` and `tests/requirements-test.txt`.
3. **Execute the tests**:
```bash
make test

```


The `Makefile` will run the tests using `pytest` with the `PYTHONPATH` set to the root directory. You can pass extra arguments using `ARGS`, for example: `make test ARGS="-v"`.

---

## Stopping the service

```bash
# Stop containers — data is preserved in Docker volumes
make down

# Stop containers AND wipe all data
make clean

```
