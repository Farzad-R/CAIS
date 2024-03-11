from pyspark.sql import SparkSession
from pyprojroot import here
spark = SparkSession.builder.appName("Test pyspark").config(
    "spark.memory.offHeap.enabled", "true").config("spark.memory.offHeap.size", "10g").getOrCreate()


df = spark.read.csv(
    str(here('data/binary/Rain/weatherAUS.csv')), header=True, escape="\"")
