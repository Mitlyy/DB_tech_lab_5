import os
import argparse
from typing import Dict, Iterable, Tuple

import yaml
from pyspark.sql import SparkSession


def _load_yaml(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(k): str(v) for k, v in data.items()}


def _parse_overrides_kv(items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--spark-conf ожидает k=v, а получили: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Пустой ключ в --spark-conf: {item}")
        out[k] = v
    return out


def _env_overrides() -> Dict[str, str]:
    raw = os.getenv("SPARK_CONF_OVERRIDES", "").strip()
    if not raw:
        return {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return _parse_overrides_kv(parts)


def build_spark_from_yaml(
    yaml_path: str = "config/spark.yaml",
    cli_overrides: Iterable[str] | None = None,
) -> SparkSession:
    base_conf = _load_yaml(yaml_path)
    env_conf = _env_overrides()
    cli_conf = _parse_overrides_kv(cli_overrides or [])

    merged: Dict[str, str] = {}
    merged.update(base_conf)
    merged.update(env_conf)
    merged.update(cli_conf)

    builder = SparkSession.builder
    for k, v in merged.items():
        builder = builder.config(k, v)

    app_name = merged.get("spark.app.name", "DBTechLab5App")
    builder = builder.appName(app_name)

    spark = builder.getOrCreate()
    return spark


def add_spark_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spark-conf",
        action="append",
        default=[],
        help="Переопределения Spark в формате k=v; можно передавать несколько раз",
    )
    parser.add_argument(
        "--spark-yaml",
        default="config/spark.yaml",
        help="Путь к YAML c настройками Spark",
    )

