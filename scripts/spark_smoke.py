import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    events_dir = os.path.abspath("logs/spark-events")
    os.makedirs(events_dir, exist_ok=True)
    events_uri = "file://" + events_dir

    spark = (
        SparkSession.builder.appName("SparkSmoke")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", events_uri)
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("INFO")

    print(f"[INFO] Spark version: {spark.version}")
    print(f"[INFO] Event log dir: {events_uri}")
    try:
        ui = spark.sparkContext.uiWebUrl
    except Exception:
        try:
            ui = spark.sparkContext._jsc.sc().uiWebUrl().get()
        except Exception:
            ui = None
    print(f"[INFO] Spark UI: {ui or 'n/a (появится при запуске джоба)'}")

    lines = spark.sparkContext.parallelize(
        ["hello spark", "hello big data", "spark makes clusters"], 4
    )
    counts = (
        lines.flatMap(lambda s: s.split())
        .map(lambda w: (w, 1))
        .reduceByKey(lambda a, b: a + b)
    )
    print("[INFO] WordCount:", counts.collect())

    df = spark.createDataFrame(
        [("A", 10), ("A", 20), ("B", 30), ("B", 50)], ["group", "value"]
    )
    agg = df.groupBy("group").agg(F.count("*").alias("n"), F.avg("value").alias("avg"))
    print("[INFO] Simple aggregation:")
    agg.show()

    print("[INFO] Done. Keep session a bit for UI…")
    spark.sql("SELECT 1").collect()

    spark.stop()


if __name__ == "__main__":
    main()
