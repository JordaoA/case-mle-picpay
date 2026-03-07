# Data Analysis — Run Guide

This folder contains the code for **Part 1** of the PicPay MLE case.

---

## Files

| File | Description |
|---|---|
| `pokemon_data_analysis` | Main Databricks notebook — ingestion, modeling, and analyses |
| `extractor.py` | PokeAPI ingestion module, imported by the notebook |

---

## How to Run on Databricks (Free Edition)

### Prerequisites

- A [Databricks Community Edition](https://community.cloud.databricks.com/) account
- The GitHub repository linked to your Databricks workspace

### Step 1 — Link the repository

1. In Databricks, go to **Repos** in the left sidebar
2. Click **Add Repo**
3. Paste the GitHub repository URL and confirm
4. Databricks will clone the repo and make it available under:
   ```
   /Workspace/Users/<your_email>/case-mle-picpay/
   ```

### Step 2 — Create a cluster

1. Go to **Compute** in the left sidebar
2. Click **Create Compute**
3. Select the following configuration:
   - **Runtime:** `14.3 LTS (Spark 3.5, Python 3.11)` or newer
   - **Node type:** Single node (sufficient for this dataset)
4. Click **Create Compute** and wait for the cluster to start

### Step 3 — Open the notebook

Navigate to the notebook in the Workspace:

```
/Workspace/Users/<your_email>/case-mle-picpay/data_analysis/pokemon_data_analysis
```

### Step 4 — Attach the cluster

At the top of the notebook, click the cluster dropdown and select the cluster you created in Step 2.

### Step 5 — Run the notebook

Click **Run All** (or use `Shift + Enter` to run cell by cell).

The notebook will execute in order:

1. **Environment setup** — installs dependencies, configures SparkSession
2. **Extraction** — fetches all ~1300 pokémons from the PokeAPI concurrently (~2–3 min)
3. **Modeling** — builds the four tables, runs data quality checks, saves as managed Delta tables
4. **Analyses** — answers Q1, Q2, and Q3 with Spark, each followed by a Seaborn plot

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

## Notes

- `extractor.py` must be in the **same folder** as the notebook so the import resolves correctly — this is already the case if you cloned the repo as described above.
- The extraction step fetches ~1300 pokémons. To run a quick smoke test, set `EXTRACT_LIMIT = 20` in the extraction cell before running.