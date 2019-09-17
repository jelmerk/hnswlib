from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only


@inherit_doc
class BruteForce(JavaEstimator):
    @keyword_only
    def __init__(self, identifierCol="id", vectorCol="vector", neighborsCol="neighbors",
                 numPartitions=1, k=5, distanceFunction="cosine"):
        super(BruteForce, self).__init__()
        self._java_obj = self._new_java_obj("com.github.jelmerk.knn.spark.bruteforce.BruteForce", self.uid)

        self.identifierCol = Param(self, "identifierCol", "the column name for the row identifier")
        self.vectorCol = Param(self, "vectorCol", "the column name for the vector")
        self.neighborsCol = Param(self, "neighborsCol", "column name for the returned neighbors")
        self.numPartitions = Param(self, "numPartitions", "number of partitions")
        self.k = Param(self, "k", "number of neighbors to find")
        self.distanceFunction = Param(self, "distanceFunction",
                                      "distance function, one of bray-curtis, canberra, cosine, correlation, euclidean, inner-product, manhattan")

        self._setDefault(identifierCol="id", vectorCol="vector", neighborsCol="neighbors", numPartitions=1, k=5,
                         distanceFunction="cosine")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, identifierCol="id", vectorCol="vector", neighborsCol="neighbors",
                  numPartitions=1, k=5, distanceFunction="cosine"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return BruteForceModel(java_model)


class BruteForceModel(JavaModel):
    """
    Model fitted by Hnsw.
    """
    def __init__(self, java_model):
        super(BruteForceModel, self).__init__(java_model)

        # note: look at https://issues.apache.org/jira/browse/SPARK-10931 in the future

        self.k = Param(self, "k", "number of neighbors to find")
        self.neighborsCol = Param(self, "neighborsCol", "column names for returned neighbors")

        self._transfer_params_from_java()