import sys
import pyspark_hnsw.evaluation

sys.modules['com.github.jelmerk.spark.knn.evaluation'] = pyspark_hnsw.evaluation
