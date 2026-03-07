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
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

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

ON_DATABRICKS = os.path.exists("/databricks")
DATABASE_NAME = "picpay_case"

if ON_DATABRICKS:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
    spark.sql(f"USE {DATABASE_NAME}")
    logger.info(f"Using Hive metastore database: {DATABASE_NAME}")

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

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Stage 2 — Data Modeling <a id="modeling"></a>
# MAGIC
# MAGIC The four tables follow the Data Dictionary from the case specification:
# MAGIC
# MAGIC | Table | PK | Description |
# MAGIC |---|---|---|
# MAGIC | `pokemon` | `pokemon_id` | Core pokémon attributes |
# MAGIC | `pokemon_type` | `(pokemon_id, type_name)` | One row per type per pokémon |
# MAGIC | `pokemon_stats` | `(pokemon_id, stat_name)` | One row per stat per pokémon |
# MAGIC | `pokemon_ability` | `(pokemon_id, ability_name)` | One row per ability per pokémon |

# COMMAND ----------

SCHEMA_POKEMON = StructType([
    StructField("pokemon_id",       IntegerType(), nullable=False),
    StructField("name",             StringType(),  nullable=False),
    StructField("height",           IntegerType(), nullable=True),
    StructField("weight",           IntegerType(), nullable=True),
    StructField("base_experience",  IntegerType(), nullable=True),
])

SCHEMA_TYPE = StructType([
    StructField("pokemon_id",  IntegerType(), nullable=False),
    StructField("type_name",   StringType(),  nullable=False),
])

SCHEMA_STATS = StructType([
    StructField("pokemon_id", IntegerType(), nullable=False),
    StructField("stat_name",  StringType(),  nullable=False),
    StructField("base_stat",  IntegerType(), nullable=False),
])

SCHEMA_ABILITY = StructType([
    StructField("pokemon_id",    IntegerType(), nullable=False),
    StructField("ability_name",  StringType(),  nullable=False),
    StructField("is_hidden",     BooleanType(), nullable=False),
])

# COMMAND ----------

df_pokemon = spark.createDataFrame(
    [asdict(p) for p in raw_data.pokemon],
    schema=SCHEMA_POKEMON,
)

df_type = spark.createDataFrame(
    [asdict(t) for t in raw_data.types],
    schema=SCHEMA_TYPE,
)

df_stats = spark.createDataFrame(
    [asdict(s) for s in raw_data.stats],
    schema=SCHEMA_STATS,
)

df_ability = spark.createDataFrame(
    [asdict(a) for a in raw_data.abilities],
    schema=SCHEMA_ABILITY,
)


# COMMAND ----------

# ---------------------------------------------------------------------------
# Data quality checks before persisting
# ---------------------------------------------------------------------------
def run_quality_checks(df_pokemon, df_type, df_stats, df_ability) -> None:
    checks = []

    # Null checks on primary keys
    checks.append((
        "pokemon: no null pokemon_id",
        df_pokemon.filter(F.col("pokemon_id").isNull()).count() == 0
    ))
    checks.append((
        "pokemon_type: no null pokemon_id or type_name",
        df_type.filter(
            F.col("pokemon_id").isNull() | F.col("type_name").isNull()
        ).count() == 0
    ))

    # Referential integrity — every type/stat/ability must link to a known pokemon
    pokemon_ids = {r.pokemon_id for r in df_pokemon.select("pokemon_id").collect()}
    type_ids    = {r.pokemon_id for r in df_type.select("pokemon_id").distinct().collect()}
    checks.append((
        "pokemon_type: all pokemon_ids exist in pokemon table",
        type_ids.issubset(pokemon_ids)
    ))

    # Each pokémon should have 6 stats
    stats_per_pokemon = (
        df_stats.groupBy("pokemon_id")
        .agg(F.count("*").alias("n_stats"))
        .filter(F.col("n_stats") != 6)
        .count()
    )
    checks.append(("pokemon_stats: each pokémon has exactly 6 stats", stats_per_pokemon == 0))

    # Results
    print("\nData Quality Report:")
    all_passed = True
    for name, passed in checks:
        status = "SUCCESS" if passed else "FAIL"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll quality checks passed!")
    else:
        raise ValueError("Data quality checks failed — review the issues above.")

run_quality_checks(df_pokemon, df_type, df_stats, df_ability)

# COMMAND ----------

def save_table(df, name: str) -> None:
    if ON_DATABRICKS:
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{DATABASE_NAME}.{name}")  # managed table — Databricks handles path
        )
        logger.info(f"Saved managed Delta table: {DATABASE_NAME}.{name} ({df.count()} rows)")
    else:
        path = f"data/processed/{name}"
        df.write.format("parquet").mode("overwrite").save(path)
        logger.info(f"Saved local parquet: {path} ({df.count()} rows)")

save_table(df_pokemon, "pokemon")
save_table(df_type,    "pokemon_type")
save_table(df_stats,   "pokemon_stats")
save_table(df_ability, "pokemon_ability")

print("\nAll tables persisted.")

# COMMAND ----------

# Register as Spark temp views for SQL queries in the analysis section
df_pokemon.createOrReplaceTempView("pokemon")
df_type.createOrReplaceTempView("pokemon_type")
df_stats.createOrReplaceTempView("pokemon_stats")
df_ability.createOrReplaceTempView("pokemon_ability")

print("Temp views registered: pokemon, pokemon_type, pokemon_stats, pokemon_ability")

# COMMAND ----------

# Quick previews
print("=== pokemon ===")
df_pokemon.show(5)

print("=== pokemon_type ===")
df_type.show(5)

print("=== pokemon_stats ===")
df_stats.show(5)

print("=== pokemon_ability ===")
df_ability.show(5)