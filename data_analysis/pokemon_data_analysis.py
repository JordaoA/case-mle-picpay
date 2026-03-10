# Databricks notebook source
# MAGIC %md
# MAGIC # PicPay ML Case — Part 1: PokeAPI Data Analysis
# MAGIC
# MAGIC **Objective:** Ingest data from the PokeAPI, model it into relational tables,
# MAGIC and answer three analytical questions using Apache Spark.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup <a id="setup"></a>

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
    ArrayType
)

from extractor import PokeAPIExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("picpay.case.part1")

sns.set_theme()
FIGSIZE = (12, 7)

try:
    spark
    logger.info("Using existing Databricks SparkSession")
except NameError:
    spark = SparkSession.builder.appName("PicPay-PokeAPI-Case").getOrCreate()
    logger.info("Created new local SparkSession")

spark.conf.set("spark.sql.shuffle.partitions", "8")

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

extractor = PokeAPIExtractor(
    page_size=100,
    max_workers=20,
    retry_attempts=3,
)

EXTRACT_LIMIT = None

logger.info(f"Starting distributed extraction — limit={EXTRACT_LIMIT or 'ALL'}")

df_raw = extractor.fetch_all_pokemon(spark, limit=EXTRACT_LIMIT)

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

pokeapi_schema = StructType([
    StructField("id", IntegerType()),
    StructField("name", StringType()),
    StructField("height", IntegerType()),
    StructField("weight", IntegerType()),
    StructField("base_experience", IntegerType()),
    StructField("types", ArrayType(StructType([
        StructField("type", StructType([
            StructField("name", StringType())
        ]))
    ]))),
    StructField("stats", ArrayType(StructType([
        StructField("base_stat", IntegerType()),
        StructField("stat", StructType([
            StructField("name", StringType())
        ]))
    ]))),
    StructField("abilities", ArrayType(StructType([
        StructField("is_hidden", BooleanType()),
        StructField("ability", StructType([
            StructField("name", StringType())
        ]))
    ])))
])

df_parsed = df_raw.withColumn("data", F.from_json(F.col("payload"), pokeapi_schema)).select("data.*")

df_parsed = df_parsed.withColumnRenamed("id", "pokemon_id")

# COMMAND ----------

df_pokemon = df_parsed.select("pokemon_id", "name", "height", "weight", "base_experience")

df_type = (
    df_parsed
    .select("pokemon_id", F.explode("types").alias("type_struct"))
    .select(
        "pokemon_id", 
        F.col("type_struct.type.name").alias("type_name")
    )
)

df_stats = (
    df_parsed
    .select("pokemon_id", F.explode("stats").alias("stat_struct"))
    .select(
        "pokemon_id", 
        F.col("stat_struct.stat.name").alias("stat_name"),
        F.col("stat_struct.base_stat").alias("base_stat")
    )
)

df_ability = (
    df_parsed
    .select("pokemon_id", F.explode("abilities").alias("ability_struct"))
    .select(
        "pokemon_id", 
        F.col("ability_struct.ability.name").alias("ability_name"),
        F.col("ability_struct.is_hidden").alias("is_hidden")
    )
)

# COMMAND ----------

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
            .saveAsTable(f"{DATABASE_NAME}.{name}")
        )
        logger.info(f"Saved managed Delta table: {DATABASE_NAME}.{name} ({df.count()} rows)")

save_table(df_pokemon, "pokemon")
save_table(df_type,    "pokemon_type")
save_table(df_stats,   "pokemon_stats")
save_table(df_ability, "pokemon_ability")

print("\nAll tables persisted.")

# COMMAND ----------

df_pokemon.createOrReplaceTempView("pokemon")
df_type.createOrReplaceTempView("pokemon_type")
df_stats.createOrReplaceTempView("pokemon_stats")
df_ability.createOrReplaceTempView("pokemon_ability")

print("Temp views registered: pokemon, pokemon_type, pokemon_stats, pokemon_ability")

# COMMAND ----------

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

df_strength = (
    df_stats
    .groupBy("pokemon_id")
    .agg(F.sum("base_stat").alias("total_strength"))
)

avg_strength = df_strength.agg(F.avg("total_strength")).collect()[0][0]
print(f"Global average strength: {avg_strength:.2f}")

df_type_count = (
    df_type
    .groupBy("pokemon_id")
    .agg(F.count("type_name").alias("num_types"))
)

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

print("  Top 15 by strength:")
(
    df_q1
    .join(df_pokemon.select("pokemon_id", "name"), on="pokemon_id")
    .orderBy(F.col("total_strength").desc())
    .select("name", "num_types", "total_strength")
    .show(15, truncate=False)
)

# COMMAND ----------

df_q1_plot = (
    df_q1
    .join(df_pokemon.select("pokemon_id", "name"), on="pokemon_id")
    .orderBy(F.col("total_strength").desc())
    .limit(15)  
    .toPandas()
)

df_q1_plot["name"] = df_q1_plot["name"].str.title()

fig, ax = plt.subplots(figsize=(10, 8))

colors = ["#ff7f0e" if i < 3 else "#8fbcd4" for i in range(len(df_q1_plot))]

sns.barplot(
    data=df_q1_plot,
    x="total_strength",
    y="name",
    palette=colors,
    ax=ax,
)

ax.axvline(
    avg_strength,
    color="#d62728",
    linestyle="--",
    linewidth=2,
    label=f"Global Average ({avg_strength:.0f})",
)

ax.set_title("Top 15 Multi-type Pokémons by Total Strength", fontsize=15, fontweight="bold", pad=20)
ax.set_xlabel("Total Strength (Sum of Base Stats)", fontsize=11)
ax.set_ylabel("") 
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2: Abilities exclusive to multi-type pokémons <a id="q2"></a>
# MAGIC
# MAGIC **Question:** Which abilities do NOT appear in any single-type pokémon?
# MAGIC In other words, abilities that exist **only** in pokémons with 2+ types.

# COMMAND ----------

df_pokemon_type_label = (
    df_type_count
    .withColumn(
        "is_multi_type",
        F.when(F.col("num_types") > 1, True).otherwise(False)
    )
)

df_ability_labeled = (
    df_ability
    .join(df_pokemon_type_label.select("pokemon_id", "is_multi_type"),
          on="pokemon_id",
          how="inner")
)

df_ability_in_single = (
    df_ability_labeled
    .filter(F.col("is_multi_type") == False)
    .select("ability_name")
    .distinct()
)

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
    .limit(15)
    .toPandas()
)

df_q2_plot["ability_name"] = df_q2_plot["ability_name"].str.replace("-", " ").str.title()

fig, ax = plt.subplots(figsize=(10, 8))

sns.barplot(
    data=df_q2_plot,
    x="pokemon_count",
    y="ability_name",
    palette="magma_r",
    ax=ax,
)

for bar in ax.patches:
    ax.text(
        bar.get_width() + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{int(bar.get_width())}",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#444444"
    )

ax.set_title("Most Common Exclusive Abilities of Multi-type Pokémons", fontsize=15, fontweight="bold", pad=20)
ax.set_xlabel("")
ax.set_ylabel("")
ax.spines[['top', 'right', 'bottom']].set_visible(False)
ax.xaxis.set_visible(False)

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

df_ability_count = (
    df_ability
    .groupBy("pokemon_id")
    .agg(F.count("ability_name").alias("num_abilities"))
)

df_versatility = (
    df_pokemon.select("pokemon_id", "name")
    .join(df_type_count,     on="pokemon_id", how="left")
    .join(df_ability_count,  on="pokemon_id", how="left")
    .join(df_strength,       on="pokemon_id", how="left")
    .fillna({"num_types": 0, "num_abilities": 0, "total_strength": 0})
)

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
    df_q3
    .withColumn("num_types_x2", F.col("num_types") * 2)
    .withColumn("num_abilities_qtd", F.col("num_abilities").cast("double"))
    .withColumn("total_strength_pct", F.col("total_strength") / 100)
    .select("name", "num_types_x2", "num_abilities_qtd", "total_strength_pct", "versatility_score")
    .toPandas()
    .sort_values(by="versatility_score", ascending=True)
)

df_q3_plot["name"] = df_q3_plot["name"].str.title()
df_q3_plot.set_index("name", inplace=True)

df_q3_plot.drop(columns=["versatility_score"], inplace=True)

colors = ["#2196F3", "#FF9800", "#4CAF50"]

ax = df_q3_plot.plot(
    kind="barh", 
    stacked=True, 
    figsize=(12, 6), 
    color=colors, 
    width=0.7,
    edgecolor="white"
)

for c in ax.containers:
    labels = [f"{v.get_width():.1f}" if v.get_width() > 0 else "" for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

ax.set_title("Top 5 Most Versatile Pokémons — Score Breakdown", fontsize=15, fontweight="bold", pad=20)
ax.set_xlabel("Versatility Score", fontsize=11)
ax.set_ylabel("")
ax.spines[['top', 'right']].set_visible(False)

ax.legend(title="Score Component", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Based on the distributed processing of the PokeAPI data using Spark, below are the consolidated answers for the three business questions proposed in the technical case:
# MAGIC
# MAGIC ### Q1: Multi-type Pokémons with above-average strength
# MAGIC **Question:** How many pokémons have more than one type and a total strength greater than the global average?
# MAGIC
# MAGIC * **Calculated Global Average Strength:** 452.10
# MAGIC * **Total Found:** **510 pokémons** meet both criteria.
# MAGIC
# MAGIC **Top 5 strongest in this category:**
# MAGIC | Rank | Pokémon | Num Types | Total Strength |
# MAGIC |---|---|:---:|:---:|
# MAGIC | 1st | Eternatus Eternamax | 2 | 1125 |
# MAGIC | 2nd | Mewtwo Mega X | 2 | 780 |
# MAGIC | 3rd | Rayquaza Mega | 2 | 780 |
# MAGIC | 4th | Zygarde Mega | 2 | 778 |
# MAGIC | 5th | Groudon Primal | 2 | 770 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q2: Abilities exclusive to multi-type pokémons
# MAGIC **Question:** Which abilities do not appear in any single-type pokémon?
# MAGIC
# MAGIC * **Total Found:** **88 abilities** are strictly exclusive to pokémons with 2 or more types.
# MAGIC
# MAGIC *Examples of exclusive abilities found in this intersection include:* `aerilate`, `air-lock`, `armor-tail`, `battery`, `battle-bond`, `commander`, `corrosion`, among others.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q3: Top 5 Most Versatile Pokémons
# MAGIC **Question:** What are the 5 most versatile pokémons, based on the formula that weights types (weight 2), abilities (weight 1), and total strength (divided by 100)?
# MAGIC
# MAGIC The data reveals that the ultimate standout in versatility is **Eternatus Eternamax**, driven mostly by its astronomical base strength, followed by variations of Kommo-o, Archaludon, and Dragapult.
# MAGIC
# MAGIC | Rank | Pokémon | Num Types | Num Abilities | Total Strength | Versatility Score |
# MAGIC |:---:|---|:---:|:---:|:---:|:---:|
# MAGIC | **1st** | **Eternatus Eternamax** | 2 | 1 | 1125 | **16.25** |
# MAGIC | **2nd** | **Kommo-o** | 2 | 3 | 600 | **13.00** |
# MAGIC | **3rd** | **Archaludon** | 2 | 3 | 600 | **13.00** |
# MAGIC | **4th** | **Dragapult** | 2 | 3 | 600 | **13.00** |
# MAGIC | **5th** | **Kommo-o Totem** | 2 | 3 | 600 | **13.00** |
