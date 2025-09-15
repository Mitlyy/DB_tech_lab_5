#!/usr/bin/env python3
import json
import os

from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ID_TEXT_COLS = ["code", "product_name", "brands", "countries"]
SAMPLE_PARQUET = "data/clean/products_sample.parquet"
FULL_PARQUET = "data/clean/products_features.parquet"
REPORT_JSON = "artifacts/kmeans/training_report.json"
CENTERS_CSV = "data/predictions/cluster_centers.csv"
PRED_OUT = "data/predictions/preds_inspect.parquet"


def pick_input_parquet():
    if os.path.isfile(SAMPLE_PARQUET):
        return SAMPLE_PARQUET
    if os.path.isfile(FULL_PARQUET):
        return FULL_PARQUET
    raise SystemExit("[FATAL] Нет входных parquet. Сначала запусти preprocess_off.py")


def pick_model_path():
    if os.path.isfile(REPORT_JSON):
        with open(REPORT_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mp = meta.get("best", {}).get("model_path")
        if mp and os.path.isdir(mp):
            return mp
    models_root = "artifacts/kmeans"
    if os.path.isdir(models_root):
        cands = [
            os.path.join(models_root, d)
            for d in os.listdir(models_root)
            if d.startswith("model_k")
        ]
        cands = [d for d in cands if os.path.isdir(d)]
        if cands:
            return sorted(cands)[-1]
    raise SystemExit("[FATAL] Не найдена модель. Сначала запусти train_kmeans.py")


def main():
    os.makedirs("data/predictions", exist_ok=True)

    spark = (
        SparkSession.builder.appName("OFF_KMeans_Inspect")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    in_path = pick_input_parquet()
    model_path = pick_model_path()
    print(f"[INFO] Using data:  {in_path}")
    print(f"[INFO] Using model: {model_path}")

    df = spark.read.parquet(in_path).cache()
    total = df.count()
    print(f"[INFO] Rows: {total}")

    if "features" not in df.columns:
        raise SystemExit(
            "[FATAL] В parquet нет колонки 'features' — нужна предобработка."
        )

    model = KMeansModel.load(model_path)
    preds = model.transform(df).cache()

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    sil = evaluator.evaluate(preds)
    print(f"[INFO] Silhouette: {sil:.5f}")

    print("\n[INFO] Cluster sizes:")
    sizes = preds.groupBy("cluster").count().orderBy("cluster")
    sizes.show(truncate=False)

    id_cols = [c for c in ID_TEXT_COLS if c in preds.columns]
    print("\n[INFO] Examples per cluster:")
    for r in sizes.collect():
        cid = r["cluster"]
        print(f"\n=== Cluster {cid} ===")
        preds.filter(F.col("cluster") == cid).select(*id_cols).limit(5).show(
            truncate=80
        )

    centers = model.clusterCenters()
    rows = []
    for i, cvec in enumerate(centers):
        row = {"cluster": i}
        for j, v in enumerate(cvec):
            row[f"f{j}"] = float(v)
        rows.append(row)
    centers_df = spark.createDataFrame(rows)
    centers_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
        CENTERS_CSV
    )

    out = preds.select(*(id_cols + ["cluster"]))
    out.write.mode("overwrite").parquet(PRED_OUT)

    print(f"\n[INFO] Saved centers CSV -> {CENTERS_CSV} (каталог с одним CSV-файлом)")
    print(f"[INFO] Saved predictions  -> {PRED_OUT}")

    spark.stop()


if __name__ == "__main__":
    main()
