# Databricks notebook source
# MAGIC %md
# MAGIC # PicPay ML Case — Part 1: PokeAPI Data Analysis
# MAGIC
# MAGIC **Objective:** Ingest data from the PokeAPI, model it into relational tables,
# MAGIC and answer three analytical questions using Apache Spark.
# MAGIC
# MAGIC **Databricks Runtime:** DBR 14.3 LTS (Spark 3.5 / Python 3.11)
# MAGIC
# MAGIC ### Table of Contents
# MAGIC 1. [Environment Setup](#setup)
# MAGIC 2. [Stage 1 — Data Extraction](#extraction)


# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Environment Setup <a id="setup"></a>

# COMMAND ----------

# Install dependencies not available in the default Databricks runtime
# (requests is already available, this is just for explicitness)
%pip install requests --quiet

# COMMAND ----------

import logging
import os
from dataclasses import asdict
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("picpay.case.part1")

try:
    spark  # noqa: F821 — injected by Databricks
    logger.info("Using existing Databricks SparkSession")
except NameError:
    spark = SparkSession.builder.appName("PicPay-PokeAPI-Case").getOrCreate()
    logger.info("Created new local SparkSession")

spark.conf.set("spark.sql.shuffle.partitions", "8")  # suitable for this dataset size

ON_DATABRICKS = os.path.exists("/dbfs")
BASE_OUTPUT_PATH = (
    "/dbfs/FileStore/picpay-case/data"
    if ON_DATABRICKS
    else "data/processed"
)
logger.info(f"Output path: {BASE_OUTPUT_PATH} | On Databricks: {ON_DATABRICKS}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Stage 1 — Data Extraction <a id="extraction"></a>
# MAGIC
# MAGIC We'll use the `PokeAPIExtractor` class, which:
# MAGIC - Paginates through `/pokemon` to collect all detail URLs
# MAGIC - Fetches each pokemon's page **concurrently** (20 workers) for performance
# MAGIC - Retries failed requests with exponential backoff
# MAGIC
# MAGIC > PokeAPI has ~1300 pokémons. Sequential fetching would take ~20 min.
# MAGIC > With 20 concurrent workers it completes in ~2–3 min.

# COMMAND ----------

# On Databricks, place extractor.py in the same folder or copy it below.
# If running from a repo connected to Databricks, this import works directly.
# Fallback: the class is also inlined at the bottom of this notebook.

try:
    from extractor import PokeAPIExtractor
    logger.info("Imported PokeAPIExtractor from extractor.py")
except ModuleNotFoundError:
    logger.warning("extractor.py not found — using inline definition below")
    # The full extractor class is defined in extractor.py
    # If you can't import it, paste the extractor.py contents here.
    raise

# COMMAND ----------

# ---------------------------------------------------------------------------
# Run extraction
# Tip: use limit=20 for a quick smoke test during development
# ---------------------------------------------------------------------------
extractor = PokeAPIExtractor(
    page_size=100,
    max_workers=20,
    retry_attempts=3,
)

# Set limit=None to fetch ALL pokémons (~1300)
# Set limit=20 for a fast test run
EXTRACT_LIMIT = 20

logger.info(f"Starting extraction — limit={EXTRACT_LIMIT or 'ALL'}")
raw_data = extractor.fetch_all_pokemon(limit=EXTRACT_LIMIT)

print(f"\n Extraction complete:")
print(f"   pokemon   : {len(raw_data.pokemon):>5} records")
print(f"   types     : {len(raw_data.types):>5} records")
print(f"   stats     : {len(raw_data.stats):>5} records")
print(f"   abilities : {len(raw_data.abilities):>5} records")