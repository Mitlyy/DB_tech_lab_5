#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("OpenFoodFactsExplore").getOrCreate()

    df = spark.read.csv(
        "data/raw/en.openfoodfacts.org.products.csv",
        header=True,
        inferSchema=True,
        sep="\t",
        multiLine=True,
        escape='"',
        quote='"',
    )

    print("\n=== SCHEMA ===")
    df.printSchema()

    print("\n=== SAMPLE 10 ROWS ===")
    df.show(10, truncate=False)

    cols = [
        "energy_100g",
        "proteins_100g",
        "fat_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "fiber_100g",
        "salt_100g",
    ]
    subset = df.select(*[c for c in cols if c in df.columns])

    print("\n=== NUTRIENTS SCHEMA ===")
    subset.printSchema()

    print("\n=== NUTRIENTS SAMPLE 10 ROWS ===")
    subset.show(10, truncate=False)

    spark.stop()
