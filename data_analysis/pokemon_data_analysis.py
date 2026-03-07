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
%pip install requests seaborn --quiet

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

sns.set_theme()
FIGSIZE = (12, 7)

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

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Stage 3 — Spark Analyses <a id="analyses"></a>

# COMMAND ----------
# MAGIC %md
# MAGIC ### Q1: Pokémons with multiple types and above-average strength <a id="q1"></a>
# MAGIC
# MAGIC **Definition of strength:** sum of all `base_stat` values for a pokémon.
# MAGIC
# MAGIC **Question:** How many pokémons have more than one `type_name`
# MAGIC and have strength greater than the overall average strength?

# COMMAND ----------

# ---------------------------------------------------------------------------
# Step 1 — Compute total strength per pokémon (sum of all base_stats)
# ---------------------------------------------------------------------------
df_strength = (
    df_stats
    .groupBy("pokemon_id")
    .agg(F.sum("base_stat").alias("total_strength"))
)

# Step 2 — Compute the global average strength (single scalar)
avg_strength = df_strength.agg(F.avg("total_strength")).collect()[0][0]
print(f"Global average strength: {avg_strength:.2f}")

# Step 3 — Count types per pokémon
df_type_count = (
    df_type
    .groupBy("pokemon_id")
    .agg(F.count("type_name").alias("num_types"))
)

# Step 4 — Join strength + type count, then filter
df_q1 = (
    df_strength
    .join(df_type_count, on="pokemon_id", how="inner")
    .filter(
        (F.col("num_types") > 1) &
        (F.col("total_strength") > avg_strength)
    )
)

answer_q1 = df_q1.count()

# COMMAND ----------

print("=" * 55)
print("  Q1 RESULT")
print("=" * 55)
print(f"\n  Pokémons with > 1 type AND strength > avg ({avg_strength:.2f}):")
print(f"\n  ➜  {answer_q1} pokémons\n")

# Show the top 10 by strength for context
print("  Top 10 by strength:")
(
    df_q1
    .join(df_pokemon.select("pokemon_id", "name"), on="pokemon_id")
    .orderBy(F.col("total_strength").desc())
    .select("name", "num_types", "total_strength")
    .show(10, truncate=False)
)

# COMMAND ----------

df_q1_plot = (
    df_q1
    .join(df_pokemon.select("pokemon_id", "name"), on="pokemon_id")
    .orderBy(F.col("total_strength").desc())
    .limit(20)
    .select("name", "total_strength")
    .toPandas()
    .sort_values("total_strength", ascending=False)
)

fig, ax = plt.subplots(figsize=FIGSIZE)

sns.barplot(
    data=df_q1_plot,
    x="name",
    y="total_strength",
    palette="Blues_d",
    ax=ax,
)

ax.axhline(
    avg_strength,
    color="crimson",
    linestyle="--",
    linewidth=1.8,
    label=f"Avg strength ({avg_strength:.0f})",
)

ax.set_title("Multi-type Pokémons Above Average Strength", fontweight="bold", pad=14)
ax.set_xlabel("Pokémon")
ax.set_ylabel("Total Strength (sum of base stats)")
ax.tick_params(axis="x")
ax.legend()

plt.tight_layout()
plt.show()


# COMMAND ----------
# MAGIC %md
# MAGIC ### Q2: Abilities exclusive to multi-type pokémons <a id="q2"></a>
# MAGIC
# MAGIC **Question:** Which abilities do NOT appear in any single-type pokémon?
# MAGIC In other words, abilities that exist **only** in pokémons with 2+ types.

# COMMAND ----------

# ---------------------------------------------------------------------------
# Step 1 — Label each pokémon as single-type or multi-type
# ---------------------------------------------------------------------------
df_pokemon_type_label = (
    df_type_count
    .withColumn(
        "is_multi_type",
        F.when(F.col("num_types") > 1, True).otherwise(False)
    )
)

# Step 2 — Join abilities with the type label
df_ability_labeled = (
    df_ability
    .join(df_pokemon_type_label.select("pokemon_id", "is_multi_type"),
          on="pokemon_id",
          how="inner")
)

# Step 3 — For each ability, check if it EVER appears in a single-type pokémon
df_ability_in_single = (
    df_ability_labeled
    .filter(F.col("is_multi_type") == False)  # noqa: E712
    .select("ability_name")
    .distinct()
)

# Step 4 — Keep only abilities NOT in the single-type set
df_q2 = (
    df_ability_labeled
    .select("ability_name")
    .distinct()
    .join(df_ability_in_single, on="ability_name", how="left_anti")
    .orderBy("ability_name")
)

answer_q2 = df_q2.count()

# COMMAND ----------

print("=" * 55)
print("  Q2 RESULT")
print("=" * 55)
print(f"\n  Abilities exclusive to multi-type pokémons: {answer_q2}\n")
df_q2.show(30, truncate=False)

# COMMAND ----------

df_q2_plot = (
    df_q2
    .join(df_ability.select("ability_name", "pokemon_id"), on="ability_name")
    .groupBy("ability_name")
    .agg(F.countDistinct("pokemon_id").alias("pokemon_count"))
    .orderBy(F.col("pokemon_count").desc())
    .limit(20)
    .toPandas()
    .sort_values("pokemon_count", ascending=True)
)

fig, ax = plt.subplots(figsize=(FIGSIZE[0], 7))

sns.barplot(
    data=df_q2_plot,
    x="pokemon_count",
    y="ability_name",
    palette="flare",
    ax=ax,
)

for bar, val in zip(ax.patches, df_q2_plot["pokemon_count"]):
    ax.text(
        bar.get_width() + 0.15,
        bar.get_y() + bar.get_height() / 2,
        str(int(val)),
        va="center",
        fontsize=10,
    )

ax.set_title(
    "Top Abilities Exclusive to Multi-type Pokémons\n(by number of pokémons that have them)",
    fontweight="bold",
    pad=14,
)
ax.set_xlabel("Number of Pokémons")
ax.set_ylabel("Ability")

plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Q3: Top 5 most versatile pokémons <a id="q3"></a>
# MAGIC
# MAGIC **Versatility score formula:**
# MAGIC ```
# MAGIC versatility_score = (num_types * 2) + (num_abilities) + (sum_stats / 100)
# MAGIC ```

# COMMAND ----------

# ---------------------------------------------------------------------------
# Step 1 — Count abilities per pokémon
# ---------------------------------------------------------------------------
df_ability_count = (
    df_ability
    .groupBy("pokemon_id")
    .agg(F.count("ability_name").alias("num_abilities"))
)

# Step 2 — Assemble all metrics per pokémon in a single DataFrame
df_versatility = (
    df_pokemon.select("pokemon_id", "name")
    .join(df_type_count,     on="pokemon_id", how="left")
    .join(df_ability_count,  on="pokemon_id", how="left")
    .join(df_strength,       on="pokemon_id", how="left")
    # Fill nulls for pokémons that might be missing some data
    .fillna({"num_types": 0, "num_abilities": 0, "total_strength": 0})
)

# Step 3 — Compute the versatility score
df_q3 = (
    df_versatility
    .withColumn(
        "versatility_score",
        (F.col("num_types") * 2)
        + F.col("num_abilities")
        + (F.col("total_strength") / 100)
    )
    .orderBy(F.col("versatility_score").desc())
    .limit(5)
    .select("name", "num_types", "num_abilities", "total_strength", "versatility_score")
)

# COMMAND ----------

print("=" * 65)
print("  Q3 RESULT — Top 5 Most Versatile Pokémons")
print("=" * 65)
df_q3.show(truncate=False)

# COMMAND ----------

df_q3_plot = (
    df_versatility
    .join(
        df_q3.select("name"),
        on="name"
    )
    .withColumn("types_component",    F.col("num_types") * 2)
    .withColumn("abilities_component", F.col("num_abilities").cast("double"))
    .withColumn("stats_component",    F.col("total_strength") / 100)
    .select("name", "types_component", "abilities_component", "stats_component")
    .toPandas()
    .sort_values(
        by=["types_component", "abilities_component", "stats_component"],
        ascending=False
    )
)

df_long = df_q3_plot.melt(
    id_vars="name",
    value_vars=["stats_component", "abilities_component", "types_component"],
    var_name="component",
    value_name="value",
)

COMPONENT_LABELS = {
    "types_component":     "Types × 2",
    "abilities_component": "Num Abilities",
    "stats_component":     "Sum Stats ÷ 100",
}
COMPONENT_COLORS = {
    "Types × 2":       "#2196F3",
    "Num Abilities":   "#FF9800",
    "Sum Stats ÷ 100": "#4CAF50",
}

df_long["component"] = df_long["component"].map(COMPONENT_LABELS)

fig, ax = plt.subplots(figsize=FIGSIZE)

bottom = pd.Series([0.0] * len(df_q3_plot), index=range(len(df_q3_plot)))
names = df_q3_plot["name"].tolist()

for component, color in COMPONENT_COLORS.items():
    values = df_long[df_long["component"] == component].set_index("name").loc[names, "value"].values
    bars = ax.bar(names, values, bottom=bottom, label=component, color=color, alpha=0.85, width=0.5)
    for bar, val, bot in zip(bars, values, bottom):
        if val > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bot + val / 2,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white",
                fontweight="bold",
            )
    bottom += values

ax.set_title("Top 5 Most Versatile Pokémons — Score Breakdown", fontweight="bold", pad=14)
ax.set_ylabel("Versatility Score")
ax.legend(title="Score Component", loc="upper right")

plt.tight_layout()
plt.show()


# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Question | Answer |
# MAGIC |---|---|
# MAGIC | **Q1** — Pokémons with >1 type AND strength > avg | See cell output |
# MAGIC | **Q2** — Abilities exclusive to multi-type pokémons | See cell output |
# MAGIC | **Q3** — Top 5 most versatile pokémons | See cell output |
# MAGIC
# MAGIC ---
# MAGIC **Data pipeline:**
# MAGIC `PokeAPI REST` → `PokeAPIExtractor (concurrent)` → `ExtractedData (dataclasses)`
# MAGIC → `Spark DataFrames` → `Quality Checks` → `Delta / Parquet`
# MAGIC → `Spark SQL Analyses`