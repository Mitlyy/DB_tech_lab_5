import os

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SAMPLED_PATH = "data/out/nutrients_sampled.parquet"
OUT_DIR = "data/out"
MODEL_DIR = os.path.join(OUT_DIR, "kmeans_model_fast")
PREDICTIONS_PATH = os.path.join(OUT_DIR, "predictions_fast.parquet")

FEATURE_COLS = [
    "energy_100g",
    "proteins_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "salt_100g",
]

K_RANGE = [3, 4, 5, 6]
MAX_ITER = 20
SIL_SAMPLE_FRACTION = 0.15


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    spark = (
        SparkSession.builder.appName("OpenFoodFactsKMeansFast")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    data = spark.read.parquet(SAMPLED_PATH)

    present_cols = [c for c in FEATURE_COLS if c in data.columns]
    if not present_cols:
        raise RuntimeError("В parquet-сэмпле нет нужных колонок.")

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
    prepared = prepared.repartition(8).cache()
    prepared.count()

    evaluator = ClusteringEvaluator(
        featuresCol="features", predictionCol="prediction", metricName="silhouette"
    )

    best_k, best_sil = None, float("-inf")
    for k in K_RANGE:
        km = KMeans(
            k=k,
            seed=42,
            featuresCol="features",
            predictionCol="prediction",
            maxIter=MAX_ITER,
            tol=1e-3,
        )
        model = km.fit(prepared)
        preds = model.transform(prepared).repartition(8).cache()

        sample = preds.sample(False, SIL_SAMPLE_FRACTION, seed=42)
        sil = evaluator.evaluate(sample)

        print(f"k={k} silhouette(sample)={sil:.5f}")
        if sil > best_sil:
            best_k, best_sil = k, sil

        preds.unpersist()

    print(f"\nЛучшее k={best_k}, silhouette(sample)={best_sil:.5f}")

    final_km = KMeans(
        k=best_k,
        seed=42,
        featuresCol="features",
        predictionCol="prediction",
        maxIter=MAX_ITER,
        tol=1e-3,
    )
    final_pipeline = Pipeline(stages=[imputer, assembler, scaler, final_km])
    final_model = final_pipeline.fit(data)

    predictions = final_model.transform(data).repartition(8).cache()
    predictions.groupBy("prediction").count().orderBy("prediction").show(truncate=False)

    sil_final = evaluator.evaluate(
        predictions.sample(False, SIL_SAMPLE_FRACTION, seed=123)
    )
    print(f"\nSilhouette(final, sample) = {sil_final:.5f}")

    final_model.write().overwrite().save(MODEL_DIR)
    predictions.select(*present_cols, "prediction").write.mode("overwrite").parquet(
        PREDICTIONS_PATH
    )

    print(f"\nМодель сохранена в: {MODEL_DIR}")
    print(f"Предсказания сохранены в: {PREDICTIONS_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
