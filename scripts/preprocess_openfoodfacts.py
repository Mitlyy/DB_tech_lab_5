#!/usr/bin/env python3
import os

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

RAW_PATH = "data/raw/en.openfoodfacts.org.products.csv.gz"
CLEAN_DIR = "data/clean"
PARQUET_OUT = os.path.join(CLEAN_DIR, "products_features.parquet")
SAMPLE_OUT = os.path.join(CLEAN_DIR, "products_sample.parquet")

FEATURE_NAME_CANDIDATES = {
    "energy": ["energy_100g", "energy-kj_100g", "energy-kcal_100g"],
    "fat": ["fat_100g", "total-fat_100g"],
    "saturated_fat": ["saturated-fat_100g", "saturated_fat_100g"],
    "carbs": ["carbohydrates_100g", "carbs_100g"],
    "sugars": ["sugars_100g"],
    "fiber": ["fiber_100g", "dietary-fiber_100g", "dietary_fiber_100g"],
    "proteins": ["proteins_100g", "protein_100g"],
    "salt": ["salt_100g"],
    "sodium": ["sodium_100g"],
}

ID_TEXT_COLS = ["code", "product_name", "brands", "countries"]


def choose_present_names(actual_cols):
    """Для каждого логического признака возвращает реально присутствующее имя колонки."""
    chosen = {}
    actual_lower = {c.lower(): c for c in actual_cols}
    for key, candidates in FEATURE_NAME_CANDIDATES.items():
        picked = None
        for cand in candidates:
            c_l = cand.lower()
            if c_l in actual_lower:
                picked = actual_lower[c_l]
                break
        if picked:
            chosen[key] = picked
    return chosen


def main():
    spark = (
        SparkSession.builder.appName("OFF_Preprocess")
        .config("spark.sql.files.maxPartitionBytes", 128 * 1024 * 1024)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df_all = (
        spark.read.option("header", True)
        .option("multiLine", False)
        .option("escape", '"')
        .csv(RAW_PATH)
    )

    print("[INFO] Total columns:", len(df_all.columns))
    print("[INFO] Sample of columns:", df_all.columns[:40])

    present_map = choose_present_names(df_all.columns)
    used_feature_cols = list(present_map.values())

    if not used_feature_cols:
        raise RuntimeError(
            "[FATAL] Не найдены числовые колонки среди ожидаемых. "
            "Проверь имена в CSV (выше распечатан список). "
            "Можно расширить FEATURE_NAME_CANDIDATES."
        )

    select_cols = [c for c in ID_TEXT_COLS if c in df_all.columns] + used_feature_cols
    df = df_all.select(*select_cols)

    for c in used_feature_cols:
        df = df.withColumn(c, F.regexp_replace(F.col(c), ",", "."))
        df = df.withColumn(c, F.col(c).cast(T.DoubleType()))

    df = df.dropna(how="all", subset=used_feature_cols)

    bounds = {
        present_map.get("energy", ""): (0.0, 8000.0),
        present_map.get("fat", ""): (0.0, 100.0),
        present_map.get("saturated_fat", ""): (0.0, 100.0),
        present_map.get("carbs", ""): (0.0, 100.0),
        present_map.get("sugars", ""): (0.0, 100.0),
        present_map.get("fiber", ""): (0.0, 100.0),
        present_map.get("proteins", ""): (0.0, 100.0),
        present_map.get("salt", ""): (0.0, 100.0),
        present_map.get("sodium", ""): (0.0, 40.0),
    }
    conditions = []
    for colname, (lo, hi) in bounds.items():
        if colname:
            conditions.append(
                (F.col(colname) >= F.lit(lo)) & (F.col(colname) <= F.lit(hi))
            )
    if conditions:
        cond_all = conditions[0]
        for z in conditions[1:]:
            cond_all = cond_all & z
        df = df.where(cond_all)

    df = df.dropna(subset=used_feature_cols)

    assembler = VectorAssembler(
        inputCols=used_feature_cols, outputCol="features_raw", handleInvalid="skip"
    )
    df_feats = assembler.transform(df)

    empty_cnt = df_feats.where(F.size("features_raw") <= 0).count()
    if empty_cnt > 0:
        raise RuntimeError(
            f"[FATAL] Обнаружены пустые векторы фичей: {empty_cnt} — проверь отбор колонок."
        )

    scaler = StandardScaler(
        withMean=True, withStd=True, inputCol="features_raw", outputCol="features"
    )
    scaler_model = scaler.fit(df_feats)
    df_scaled = scaler_model.transform(df_feats).drop("features_raw")

    os.makedirs(CLEAN_DIR, exist_ok=True)
    df_scaled.write.mode("overwrite").parquet(PARQUET_OUT)

    sample_n = 200_000
    total = df_scaled.count()
    frac = min(1.0, sample_n / max(1, total))
    df_sample = df_scaled.sample(withReplacement=False, fraction=frac, seed=42)
    df_sample.write.mode("overwrite").parquet(SAMPLE_OUT)

    print(f"[INFO] Cleaned full dataset -> {PARQUET_OUT}")
    print(f"[INFO] Sample for fast iteration -> {SAMPLE_OUT}")
    print(f"[INFO] Rows full: {total}, rows sample: {df_sample.count()}")
    print(f"[INFO] Used feature columns: {used_feature_cols}")
    print(f"[INFO] Feature logical mapping: {present_map}")

    spark.stop()


if __name__ == "__main__":
    main()
