from pyspark.ml.evaluation import JavaEvaluator
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable

__all__ = ['KnnSimilarityEvaluator']

@inherit_doc
class KnnSimilarityEvaluator(JavaEvaluator, JavaMLReadable, JavaMLWritable):
    """
    Evaluate the performance of a knn model.
    """
    @keyword_only
    def __init__(self, approximateNeighborsCol="approximateNeighbors", exactNeighborsCol="exactNeighbors"):
        super(JavaEvaluator, self).__init__()
        self._java_obj = self._new_java_obj("com.github.jelmerk.spark.knn.evaluation.KnnSimilarityEvaluator", self.uid)

        self.approximateNeighborsCol = Param(self, "approximateNeighborsCol", "the column name for the row identifier")
        self.exactNeighborsCol = Param(self, "exactNeighborsCol", "the column name for the vector")

        self._setDefault(approximateNeighborsCol="approximateNeighbors", exactNeighborsCol="exactNeighbors")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def getApproximateNeighborsCol(self):
        """
        Gets the value of approximateNeighborsCol or its default value.
        """
        return self.getOrDefault(self.approximateNeighborsCol)

    def setApproximateNeighborsCol(self, value):
        """
        Sets the value of :py:attr:`approximateNeighborsCol`.
        """
        return self._set(approximateNeighborsCol=value)

    def getExactNeighborsCol(self):
        """
        Gets the value of exactNeighborsCol or its default value.
        """
        return self.getOrDefault(self.exactNeighborsCol)

    def setExactNeighborsCol(self, value):
        """
        Sets the value of :py:attr:`exactNeighborsCol`.
        """
        return self._set(exactNeighborsCol=value)

    @keyword_only
    def setParams(self, approximateNeighborsCol="approximateNeighbors", exactNeighborsCol="exactNeighbors"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
