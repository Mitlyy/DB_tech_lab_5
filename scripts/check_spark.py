import os
import platform
import subprocess
import sys

from pyspark.sql import SparkSession


def java_version():
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return (out.stderr or out.stdout).strip().splitlines()[0]
    except Exception as e:
        return f"java -version error: {e}"


def main():
    spark = SparkSession.builder.appName("SparkEnvCheck").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("INFO")

    print("[INFO] Python:", platform.python_version())
    try:
        import pyspark

        print("[INFO] PySpark:", pyspark.__version__)
    except Exception as e:
        print("[INFO] PySpark version read error:", e)

    print("[INFO] Spark:", spark.version)
    print("[INFO] Java:", java_version())
    print("[INFO] Master:", sc.master)

    ui = None
    try:
        ui = sc.uiWebUrl
    except Exception:
        try:
            ui = sc._jsc.sc().uiWebUrl().get()
        except Exception:
            ui = None
    print("[INFO] Spark UI:", ui or "n/a")

    print("[INFO] Selected Spark conf:")
    for k in sorted(
        set(
            [
                "spark.eventLog.enabled",
                "spark.eventLog.dir",
                "spark.sql.shuffle.partitions",
                "spark.driver.memory",
                "spark.executor.memory",
                "spark.default.parallelism",
            ]
        )
    ):
        print(f"  - {k} = {sc.getConf().get(k, 'unset')}")

    spark.stop()


if __name__ == "__main__":
    main()
