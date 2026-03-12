# PicPay — Machine Learning Engineer Case

This repository contains the technical case solution for the **Machine Learning Engineer** position at PicPay.

The challenge is split into two independent parts:

---

## Part 1 — Data Analysis with PokeAPI

The goal of this part is to demonstrate skills in REST API ingestion, relational data modeling, and data manipulation using Apache Spark.

Data is consumed from the public [PokeAPI](https://pokeapi.co/docs/v2#pokemon-section), which provides detailed information about pokémons. From each pokémon's detail page, three specific datasets are extracted — types, stats, and abilities — and modeled into four relational tables:

| Table | Description |
| --- | --- |
| `pokemon` | Core attributes: id, name, height, weight, base experience |
| `pokemon_type` | One row per type per pokémon (e.g. fire, water) |
| `pokemon_stats` | One row per stat per pokémon (hp, attack, defense, etc.) |
| `pokemon_ability` | One row per ability per pokémon, with hidden flag |

With those tables in place, three analytical questions are answered using Spark:

* **Q1 —** How many pokémons have more than one type and a total strength above the overall average?
* **Q2 —** Which abilities appear exclusively in multi-type pokémons (i.e. never in a single-type one)?
* **Q3 —** What are the top 5 most versatile pokémons, scored by `(num_types × 2) + num_abilities + (sum_stats ÷ 100)`?

Each question is accompanied by a Seaborn visualization in the notebook.

→ **[How to run — Data Analysis guide](data_analysis/README.md)**

---

## Part 2 — NER Inference Microservice

A production-ready REST microservice for **Named Entity Recognition (NER)**, built with FastAPI, spaCy, MLflow, and MongoDB — fully containerized with Docker Compose.

### What it does

The service accepts free-form English text, runs NER inference using a loaded spaCy model, and returns the recognized entities (people, organizations, dates, monetary values, etc.) with their character offsets. Every prediction is logged to MLflow as an experiment run and persisted to MongoDB for history retrieval.

### Stack

| Component | Role |
| --- | --- |
| **FastAPI** | REST API framework |
| **spaCy** | NER inference engine |
| **MLflow** | Model registry + experiment tracking |
| **MongoDB** | Persistent prediction history |
| **Docker Compose** | Orchestrates all three services |

### API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/load/` | Download a spaCy model and register it in MLflow |
| `POST` | `/predict/` | Run NER inference on text, log result to MLflow + MongoDB |
| `GET` | `/models/` | List all registered models with registry metadata |
| `DELETE` | `/models/{name}` | Archive a model in MLflow and evict it from cache |
| `GET` | `/list/` | Retrieve full prediction history |
| `GET` | `/health/` | Service health, loaded models, and MongoDB status |

### Example flow

```bash
# 1. Start the stack
make up

# 2. Load a model
curl -X POST http://localhost:8000/load/ \
  -H "Content-Type: application/json" \
  -d '{"model": "en_core_web_sm"}'

# 3. Run a prediction
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple acquired Beats for $3 billion in 2014.", "model": "en_core_web_sm"}'

# 4. Browse results in MLflow
open http://localhost:5000

```

### Design highlights

* **Lazy model cache** — spaCy models are loaded into memory on first use and kept in a thread-safe in-process cache.
* **MLflow integration** — every `/load/` call registers a model version; every `/predict/` call logs a run with metadata like entity counts and label distributions.
* **Persistence** — uses MongoDB to store a durable history of all predictions made through the service.
* **Fully containerized** — a single `make up` builds and starts all three services (API, MLflow, MongoDB) with health-checked startup ordering.

→ **[How to run — Microservice guide](microservice/README.md)**

---

## Repository Structure

```
case-mle-picpay/
├── README.md                    # This file
├── data_analysis/
│   ├── README.md                # Run guide for Part 1
│   ├── pokemon_data_analysis    # Main Databricks notebook
│   └── extractor.py             # PokeAPI ingestion module
└── microservice/
    ├── README.md                # Run guide for Part 2
    ├── app/                     # FastAPI application
    ├── Dockerfile               # Microservice container definition
    ├── docker-compose.yaml      # Multi-container orchestration
    ├── Makefile                 # Command automation
    └── ...

```
