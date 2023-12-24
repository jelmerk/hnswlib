import sys
import pyspark_hnsw.conversion

sys.modules['com.github.jelmerk.spark.conversion'] = pyspark_hnsw.conversion
