#!/usr/bin/env python3
import gzip
import os
import sys
from contextlib import closing

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

RAW_TSV = "data/raw/en.openfoodfacts.org.products.csv.gz"
RAW_JSONL = "data/raw/openfoodfacts-products.jsonl.gz"

FEATURE_NAME_CANDIDATES = {
    "energy": ["energy-kcal_100g", "energy-kj_100g", "energy_100g"],
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

DO_COUNT = False


def file_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def gzip_valid(path: str, test_bytes: int = 1024 * 1024) -> bool:
    """Быстрая проверка целостности .gz (ловим обрыв EOF)."""
    if not file_ok(path):
        return False
    try:
        with closing(gzip.open(path, "rb")) as f:
            _ = f.read(test_bytes)
        return True
    except Exception as e:
        print(f"[WARN] gzip_valid failed for {path}: {e}", file=sys.stderr)
        return False


def choose_source() -> str | None:
    """
    Предпочтение TSV (быстрее для обзора). Если валидный TSV есть — используем его,
    иначе пробуем JSONL. Если обоих нет/битые — None.
    """
    if file_ok(RAW_TSV) and gzip_valid(RAW_TSV):
        return "tsv"
    if file_ok(RAW_JSONL) and gzip_valid(RAW_JSONL):
        return "jsonl"
    return None


def read_tsv(spark, path):
    return (
        spark.read.option("header", True)
        .option("sep", "\t")
        .option("multiLine", False)
        .csv(path)
    )


def read_jsonl(spark, path):
    df = spark.read.option("multiLine", False).json(path)
    if "nutriments" in df.columns:
        for _, cands in FEATURE_NAME_CANDIDATES.items():
            for candidate in cands:
                jkey = candidate.replace("-", "_")
                df = df.withColumn(candidate, F.col(f"nutriments.{jkey}"))
    return df


def pick_present_numeric_cols(all_cols):
    """Выбираем реально существующие числовые колонки из карты синонимов."""
    lower2orig = {c.lower(): c for c in all_cols}
    present = []
    logical2actual = {}
    for logical, cands in FEATURE_NAME_CANDIDATES.items():
        for cand in cands:
            c_l = cand.lower()
            if c_l in lower2orig:
                present.append(lower2orig[c_l])
                logical2actual[logical] = lower2orig[c_l]
                break
    return present, logical2actual


def main():
    spark = (
        SparkSession.builder.appName("OFF_Peek")
        .config("spark.sql.files.maxPartitionBytes", 128 * 1024 * 1024)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    src = choose_source()
    if src is None:
        print("[FATAL] Нет валидных данных. Сначала скачай:", file=sys.stderr)
        print("  ./scripts/download_off.sh csv   # TSV", file=sys.stderr)
        print("  ./scripts/download_off.sh jsonl # JSONL", file=sys.stderr)
        sys.exit(1)

    if src == "tsv":
        print("[INFO] Using TSV export.")
        df = read_tsv(spark, RAW_TSV)
        source_path = RAW_TSV
    else:
        print("[INFO] Using JSONL dump.")
        df = read_jsonl(spark, RAW_JSONL)
        source_path = RAW_JSONL

    print(f"[INFO] Source file: {source_path}")
    print(f"[INFO] Columns total: {len(df.columns)}")
    print("[INFO] First 50 column names:")
    for c in df.columns[:50]:
        print("  -", c)

    print("\n[INFO] Schema:")
    df.printSchema()

    print("\n[INFO] Head (20 rows):")
    df.limit(20).show(truncate=80)

    if DO_COUNT:
        print("\n[INFO] Counting rows (this may take a while)...")
        total = df.count()
        print(f"[INFO] Total rows: {total}")

    present_numeric, mapping = pick_present_numeric_cols(df.columns)
    print("\n[INFO] Present numeric nutrient columns:")
    for c in present_numeric:
        print("  -", c)
    print("[INFO] Logical→actual mapping:", mapping)

    if present_numeric:
        for c in present_numeric:
            df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), ",", "."))
            df = df.withColumn(c, F.col(c).cast(T.DoubleType()))

        print("\n[INFO] describe() for numeric nutrients:")
        df.select(*present_numeric).describe().show(truncate=False)

        print("\n[INFO] Null counts for numeric nutrients:")
        null_exprs = [
            F.sum(F.col(c).isNull().cast("int")).alias(c) for c in present_numeric
        ]
        df.select(*null_exprs).show(truncate=False)
    else:
        print(
            "\n[WARN] Не найдено ни одной ожидаемой числовой колонки нутриентов. "
            "Скорректируй FEATURE_NAME_CANDIDATES под свою выгрузку."
        )

    idtext = [c for c in ID_TEXT_COLS if c in df.columns]
    if idtext:
        print("\n[INFO] Sample of ID/TEXT columns:")
        df.select(*idtext).limit(10).show(truncate=80)

    spark.stop()


if __name__ == "__main__":
    main()
