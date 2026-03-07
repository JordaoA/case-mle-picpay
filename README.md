# PicPay — Machine Learning Engineer Case

This repository contains the technical case solution for the **Machine Learning Engineer** position at PicPay.

The challenge is split into two independent parts:

---

## Part 1 — Data Analysis with PokeAPI

The goal of this part is to demonstrate skills in REST API ingestion, relational data modeling, and data manipulation using Apache Spark.

Data is consumed from the public [PokeAPI](https://pokeapi.co/api/v2/pokemon), which provides detailed information about pokémons. From each pokémon's detail page, three specific datasets are extracted — types, stats, and abilities — and modeled into four relational tables:

| Table | Description |
|---|---|
| `pokemon` | Core attributes: id, name, height, weight, base experience |
| `pokemon_type` | One row per type per pokémon (e.g. fire, water) |
| `pokemon_stats` | One row per stat per pokémon (hp, attack, defense, etc.) |
| `pokemon_ability` | One row per ability per pokémon, with hidden flag |

With those tables in place, three analytical questions are answered using Spark:

- **Q1 —** How many pokémons have more than one type and a total strength above the overall average?
- **Q2 —** Which abilities appear exclusively in multi-type pokémons (i.e. never in a single-type one)?
- **Q3 —** What are the top 5 most versatile pokémons, scored by `(num_types × 2) + num_abilities + (sum_stats ÷ 100)`?

Each question is accompanied by a Seaborn visualization in the notebook.

→ **[How to run — Data Analysis guide](data_analysis/README.md)**