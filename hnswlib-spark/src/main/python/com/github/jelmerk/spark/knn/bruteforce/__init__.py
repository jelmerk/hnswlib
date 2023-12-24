import sys
import pyspark_hnsw.knn

sys.modules['com.github.jelmerk.spark.knn.bruteforce'] = pyspark_hnsw.knn
