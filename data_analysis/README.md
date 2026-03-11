# Data Analysis — Run Guide

This folder contains the code for **Part 1** of the PicPay MLE case.

---

## Files

| File | Description |
|---|---|
| `pokemon_data_analysis` | Main Databricks notebook — distributed ingestion, modeling, and Spark SQL analyses. |
| `extractor.py` | PokeAPI ingestion module. Optimized for serverless/Spark environments using `mapInPandas` for distributed HTTP requests. |

---

## How to Run on Databricks (Free Edition)

### Prerequisites

- A [Databricks Community Edition](https://community.cloud.databricks.com/) account  
- The GitHub repository linked to your Databricks workspace  

---

## Step 1 — Link the repository

1. In Databricks, click `+ New` in the left sidebar  
2. Go to `More > Git folder`  
3. Paste the GitHub repository URL and confirm  
4. Databricks will clone the repo and make it available under:

```text
/Workspace/Users/<your-display-name>/case-mle-picpay/
```

---

## Step 2 — Open the notebook

Navigate to the notebook in the Workspace:

```text
/Workspace/Users/<your-display-name>/case-mle-picpay/data_analysis/pokemon_data_analysis
```

---

## Step 3 — Run the notebook

Click **Run All** (or use `Shift + Enter` to run cell by cell).

The notebook will execute in order:

### 1. Environment setup
Installs dependencies (`requests`, `seaborn`) and configures the `SparkSession`.

### 2. Extraction
Fetches all ~1300 pokémons from the **PokeAPI**. This step is fully distributed across Spark workers using `mapInPandas`, optimizing network I/O and connection pooling.

### 3. Modeling
Parses the raw JSON payloads distributively using Spark's Catalyst Optimizer (`from_json`, `explode`), builds the four relational tables, runs data quality checks, and saves them as managed Delta tables.

### 4. Analyses
Answers Q1, Q2, and Q3 using Spark DataFrames, each followed by an enhanced Seaborn plot.

---

## Expected Output

After a successful run, four managed Delta tables will be available in the Hive metastore under the `picpay_case` database:

```sql
SELECT * FROM picpay_case.pokemon LIMIT 10;
SELECT * FROM picpay_case.pokemon_type LIMIT 10;
SELECT * FROM picpay_case.pokemon_stats LIMIT 10;
SELECT * FROM picpay_case.pokemon_ability LIMIT 10;
```

You can query them directly from any `%sql` cell in Databricks after the notebook completes.

---
