from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("WordCount").getOrCreate()

data = ["hello world", "hello spark", "big data spark"]
rdd = spark.sparkContext.parallelize(data)

word_counts = (
    rdd.flatMap(lambda line: line.split(" "))
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
)

print(word_counts.collect())
spark.stop()
