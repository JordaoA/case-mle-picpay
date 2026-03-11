SHELL        := /bin/bash
.SHELLFLAGS  := -euo pipefail -c

ROOT_DIR  := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TESTS_DIR := $(ROOT_DIR)/tests
REQ_APP   := $(ROOT_DIR)/requirements.txt
REQ_TEST  := $(TESTS_DIR)/requirements-test.txt

COMPOSE := docker compose -f $(ROOT_DIR)/docker-compose.yaml

.DEFAULT_GOAL := help

ARGS ?=

.PHONY: help
help:
	@printf "\n"
	@printf "  \033[1mNER Inference Service\033[0m\n"
	@printf "\n"
	@printf "  \033[36mmake install\033[0m     Install all dependencies on the host\n"
	@printf "  \033[36mmake up\033[0m          Build and start the full stack in Docker\n"
	@printf "  \033[36mmake down\033[0m        Stop containers, keep volumes\n"
	@printf "  \033[36mmake clean\033[0m       Stop containers AND delete all volumes\n"
	@printf "  \033[36mmake logs\033[0m        Tail container logs (Ctrl+C to stop)\n"
	@printf "  \033[36mmake test\033[0m        Run all tests (unit & integration)\n"
	@printf "  \033[36mmake test-unit\033[0m   Run only unit tests\n"
	@printf "  \033[36mmake test-int\033[0m    Run only integration tests\n"
	@printf "\n"

.PHONY: install
install:
	@printf "\n\033[1m── Installing microservice dependencies ──\033[0m\n"
	pip install --upgrade pip
	pip install -r $(REQ_APP)
	@printf "\n\033[1m── Installing test dependencies ──\033[0m\n"
	pip install -r $(REQ_TEST)
	@printf "\n\033[32m✔ All dependencies installed.\033[0m\n\n"

.PHONY: up
up: _ensure-env
	@printf "\n\033[1m── Building and starting full stack ──\033[0m\n"
	$(COMPOSE) up --build -d
	@printf "\n\033[1m── Waiting for services to be healthy ──\033[0m\n"
	@$(MAKE) _wait-healthy SVC=picpay-mongo
	@$(MAKE) _wait-healthy SVC=picpay-mlflow
	@$(MAKE) _wait-healthy SVC=picpay-ner-service
	@printf "\n\033[32m✔ Stack is up and healthy.\033[0m\n\n"
	@printf "   API        → http://localhost:8000\n"
	@printf "   Swagger UI → http://localhost:8000/docs\n"
	@printf "   MLflow UI  → http://localhost:5000\n\n"
	@printf "   Run \033[36mmake logs\033[0m to tail output.\n"
	@printf "   Run \033[36mmake down\033[0m to stop.\n\n"

.PHONY: down
down:
	@printf "\033[1m── Stopping containers (volumes preserved) ──\033[0m\n"
	-$(COMPOSE) down
	@printf "\033[32m✔ Containers stopped.\033[0m\n"

.PHONY: clean
clean:
	@printf "\033[1m── Stopping containers and deleting volumes ──\033[0m\n"
	@printf "   WARNING: This will delete all MLflow and MongoDB data.\n"
	-$(COMPOSE) down -v
	@printf "\033[32m✔ Containers and volumes removed.\033[0m\n"

.PHONY: logs
logs:
	@printf "\033[1m── Container logs (Ctrl+C to stop) ──\033[0m\n"
	$(COMPOSE) logs -f

.PHONY: test-unit
test-unit: _ensure-install
	@printf "\n\033[1m── Running unit tests ──\033[0m\n\n"
	cd $(ROOT_DIR) && \
	PYTHONPATH=$(ROOT_DIR) \
	pytest $(TESTS_DIR)/unit/ \
	    --tb=short \
	    -q \
	    $(ARGS)
	@printf "\n\033[32m✔ Unit tests passed.\033[0m\n\n"

.PHONY: test-int
test-int: _ensure-install
	@printf "\n\033[1m── Running integration tests ──\033[0m\n\n"
	cd $(ROOT_DIR) && \
	PYTHONPATH=$(ROOT_DIR) \
	pytest $(TESTS_DIR)/integration/ \
	    --tb=short \
	    -q \
	    $(ARGS)
	@printf "\n\033[32m✔ Integration tests passed.\033[0m\n\n"

.PHONY: test
test: test-unit test-int
	@printf "\n\033[32m✔ All test suites passed.\033[0m\n\n"

.PHONY: _ensure-env
_ensure-env:
	@if [ ! -f $(ROOT_DIR)/.env ]; then \
	    printf "\033[1m── No .env found — copying from .env.example ──\033[0m\n"; \
	    cp $(ROOT_DIR)/.env.example $(ROOT_DIR)/.env; \
	    printf "   Edit .env to customise settings.\n\n"; \
	fi

.PHONY: _ensure-install
_ensure-install:
	@python3 -c "import fastapi, uvicorn, spacy, mlflow, pymongo" 2>/dev/null || \
	    { printf "\033[1mDependencies missing — running make install first.\033[0m\n"; \
	      $(MAKE) -C $(ROOT_DIR) install; }

.PHONY: _wait-healthy
_wait-healthy:
	@printf "   Waiting for $(SVC) …\n"; \
	for i in $$(seq 1 30); do \
	    STATUS=$$(docker inspect --format='{{.State.Health.Status}}' $(SVC) 2>/dev/null || printf "missing"); \
	    if [ "$$STATUS" = "healthy" ]; then \
	        printf "   \033[32m✔ $(SVC) is healthy.\033[0m\n"; \
	        exit 0; \
	    fi; \
	    sleep 2; \
	done; \
	printf "   $(SVC) did not become healthy in time.\n"; \
	docker inspect --format='{{json .State.Health}}' $(SVC); \
	exit 1