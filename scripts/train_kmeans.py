import os

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

DATA_PATH = "data/raw/en.openfoodfacts.org.products.csv"
OUT_DIR = "data/out"
MODEL_DIR = os.path.join(OUT_DIR, "kmeans_model")
PREDICTIONS_PATH = os.path.join(OUT_DIR, "predictions.parquet")
SAMPLED_PATH = os.path.join(OUT_DIR, "nutrients_sampled.parquet")
BAD_RECORDS_DIR = os.path.join(OUT_DIR, "bad_records_csv")

FEATURE_COLS = [
    "energy_100g",
    "proteins_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "salt_100g",
]

MAX_ROWS = 200_000
K_RANGE = list(range(3, 11))


def safe_to_double(colname: str):
    s = F.col(colname).cast("string")
    s = F.regexp_replace(s, ",", ".")
    extracted = F.regexp_extract(s, r"(-?\d+(?:\.\d+)?)", 0)
    return F.when(F.length(extracted) > 0, extracted.cast(DoubleType())).otherwise(
        F.lit(None).cast(DoubleType())
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    spark = SparkSession.builder.appName("OpenFoodFactsKMeans").getOrCreate()

    df = (
        spark.read.option("header", True)
        .option("sep", "\t")
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .option("badRecordsPath", BAD_RECORDS_DIR)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(DATA_PATH)
    )

    present_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not present_cols:
        raise RuntimeError(
            "В CSV нет ни одной из ожидаемых nutrient-колонок: "
            + ", ".join(FEATURE_COLS)
        )

    data = df.select(*present_cols)
    for c in present_cols:
        data = data.withColumn(c, safe_to_double(c))

    limits = {
        "energy_100g": (0.0, 4000.0),
        "proteins_100g": (0.0, 100.0),
        "fat_100g": (0.0, 100.0),
        "carbohydrates_100g": (0.0, 100.0),
        "sugars_100g": (0.0, 100.0),
        "fiber_100g": (0.0, 100.0),
        "salt_100g": (0.0, 100.0),
    }
    for c in present_cols:
        lo, hi = limits[c]
        data = data.filter((F.col(c).isNull()) | (F.col(c) >= F.lit(lo)))
        data = data.filter((F.col(c).isNull()) | (F.col(c) <= F.lit(hi)))

    not_all_null = None
    for c in present_cols:
        cond = F.col(c).isNotNull()
        not_all_null = cond if not_all_null is None else (not_all_null | cond)
    data = data.filter(not_all_null)

    approx_count = data.count()
    if approx_count > MAX_ROWS:
        frac = MAX_ROWS / float(approx_count)
        data = data.sample(withReplacement=False, fraction=frac, seed=42).limit(
            MAX_ROWS
        )

    data.write.mode("overwrite").parquet(SAMPLED_PATH)

    imputer = Imputer(
        strategy="median", inputCols=present_cols, outputCols=present_cols
    )
    assembler = VectorAssembler(inputCols=present_cols, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True
    )

    base_pipeline = Pipeline(stages=[imputer, assembler, scaler])
    prepared_model = base_pipeline.fit(data)
    prepared = prepared_model.transform(data).select("features")

    evaluator = ClusteringEvaluator(
        featuresCol="features", predictionCol="prediction", metricName="silhouette"
    )

    best_k, best_silhouette, best_model = None, float("-inf"), None
    for k in K_RANGE:
        km = KMeans(
            k=k, seed=42, featuresCol="features", predictionCol="prediction", maxIter=50
        )
        model = km.fit(prepared)
        preds = model.transform(prepared)
        sil = evaluator.evaluate(preds)
        print(f"k={k} silhouette={sil:.5f}")
        if sil > best_silhouette:
            best_k = k
            best_silhouette = sil
            best_model = model

    print(f"\nЛучшее k={best_k}, silhouette={best_silhouette:.5f}")

    final_km = KMeans(
        k=best_k,
        seed=42,
        featuresCol="features",
        predictionCol="prediction",
        maxIter=50,
    )
    final_pipeline = Pipeline(stages=[imputer, assembler, scaler, final_km])
    final_pipeline_model = final_pipeline.fit(data)

    predictions = final_pipeline_model.transform(data)
    print("\n=== Cluster counts ===")
    predictions.groupBy("prediction").count().orderBy("prediction").show(truncate=False)

    silhouette_final = evaluator.evaluate(predictions)
    print(f"\nSilhouette(final) = {silhouette_final:.5f}")

    km_stage = final_pipeline_model.stages[-1]
    centers = km_stage.clusterCenters()
    print("\n=== Cluster centers (standardized space) ===")
    for i, c in enumerate(centers):
        print(f"Cluster {i}: {c}")

    final_pipeline_model.write().overwrite().save(MODEL_DIR)
    predictions.select(*present_cols, "prediction").write.mode("overwrite").parquet(
        PREDICTIONS_PATH
    )

    print(f"\nМодель сохранена в: {MODEL_DIR}")
    print(f"Предсказания сохранены в: {PREDICTIONS_PATH}")
    print(f"Проблемные строки (если были) — в каталоге: {BAD_RECORDS_DIR}")

    spark.stop()


if __name__ == "__main__":
    main()
