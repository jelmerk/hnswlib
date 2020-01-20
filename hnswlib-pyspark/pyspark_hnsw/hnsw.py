from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable

__all__ = ['Hnsw', 'HnswModel']

@inherit_doc
class Hnsw(JavaEstimator, JavaMLReadable, JavaMLWritable):
    """
    Approximate nearest neighbour search.
    """

    @keyword_only
    def __init__(self, identifierCol="id", vectorCol="vector", neighborsCol="neighbors",
                 m=16, ef=10, efConstruction=200, numPartitions=1, k=5, distanceFunction="cosine",
                 excludeSelf=False, similarityThreshold=-1.0, outputFormat="full"):
        super(Hnsw, self).__init__()
        self._java_obj = self._new_java_obj("com.github.jelmerk.spark.knn.hnsw.Hnsw", self.uid)

        self.identifierCol = Param(self, "identifierCol", "the column name for the row identifier")
        self.vectorCol = Param(self, "vectorCol", "the column name for the vector")
        self.neighborsCol = Param(self, "neighborsCol", "column name for the returned neighbors")
        self.m = Param(self, "m", "number of bi-directional links created for every new element during construction")
        self.ef = Param(self, "ef", "size of the dynamic list for the nearest neighbors (used during the search)")
        self.efConstruction = Param(self, "efConstruction",
                                    "has the same meaning as ef, but controls the index time / index precision")
        self.numPartitions = Param(self, "numPartitions", "number of partitions")
        self.k = Param(self, "k", "number of neighbors to find")
        self.distanceFunction = Param(self, "distanceFunction",
                                      "distance function, one of bray-curtis, canberra, cosine, correlation, "
                                      "euclidean, inner-product, manhattan or the fully qualified classname "
                                      "of a distance function")
        self.excludeSelf = Param(self, "excludeSelf", "whether to include the row identifier as a candidate neighbor")
        self.similarityThreshold = Param(self, "similarityThreshold",
                                         "do not return neighbors further away than this distance")
        self.outputFormat = Param(self, "outputFormat", "output format, one of full, minimal")
        self.storageLevel = Param(self, "storageLevel",
                                  "storageLevel for the indices. Pass in a string representation of StorageLevel")

        self._setDefault(identifierCol="id", vectorCol="vector", neighborsCol="neighbors",
                         m=16, ef=10, efConstruction=200, numPartitions=1, k=5, distanceFunction="cosine",
                         excludeSelf=False, similarityThreshold=-1.0, outputFormat="full", storageLevel="MEMORY_ONLY")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)


    def getIdentifierCol(self):
        """
        Gets the value of identifierCol or its default value.
        """
        return self.getOrDefault(self.identifierCol)

    def setIdentifierCol(self, value):
        """
        Sets the value of :py:attr:`identifierCol`.
        """
        return self._set(identifierCol=value)

    def getVectorCol(self):
        """
        Gets the value of vectorCol or its default value.
        """
        return self.getOrDefault(self.vectorCol)

    def setVectorCol(self, value):
        """
        Sets the value of :py:attr:`vectorCol`.
        """
        return self._set(vectorCol=value)

    def getNeighborsCol(self):
        """
        Gets the value of neighborsCol or its default value.
        """
        return self.getOrDefault(self.neighborsCol)

    def setNeighborsCol(self, value):
        """
        Sets the value of :py:attr:`neighborsCol`.
        """
        return self._set(neighborsCol=value)

    def getNumPartitions(self):
        """
        Gets the value of numPartitions or its default value.
        """
        return self.getOrDefault(self.numPartitions)

    def setNumPartitions(self, value):
        """
        Sets the value of :py:attr:`numPartitions`.
        """
        return self._set(numPartitions=value)

    def getK(self):
        """
        Gets the value of k or its default value.
        """
        return self.getOrDefault(self.k)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set(k=value)

    def getDistanceFunction(self):
        """
        Gets the value of distanceFunction or its default value.
        """
        return self.getOrDefault(self.distanceFunction)

    def setDistanceFunction(self, value):
        """
        Sets the value of :py:attr:`distanceFunction`.
        """
        return self._set(distanceFunction=value)

    def getExcludeSelf(self):
        """
        Gets the value of excludeSelf or its default value.
        """
        return self.getOrDefault(self.excludeSelf)

    def setExcludeSelf(self, value):
        """
        Sets the value of :py:attr:`excludeSelf`.
        """
        return self._set(excludeSelf=value)

    def getSimilarityThreshold(self):
        """
        Gets the value of similarityThreshold or its default value.
        """
        return self.getOrDefault(self.similarityThreshold)

    def setSimilarityThreshold(self, value):
        """
        Sets the value of :py:attr:`similarityThreshold`.
        """
        return self._set(similarityThreshold=value)

    def getOutputFormat(self):
        """
        Gets the value of outputFormat or its default value.
        """
        return self.getOrDefault(self.outputFormat)

    def setOutputFormat(self, value):
        """
        Sets the value of :py:attr:`outputFormat`.
        """
        return self._set(outputFormat=value)

    def getStorageLevel(self):
        """
        Gets the value of storageLevel or its default value.
        """
        return self.getOrDefault(self.storageLevel)

    def setStorageLevel(self, value):
        """
        Sets the value of :py:attr:`storageLevel`.
        """
        return self._set(storageLevel=value)

    def getM(self):
        """
        Gets the value of m or its default value.
        """
        return self.getOrDefault(self.m)

    def setM(self, value):
        """
        Sets the value of :py:attr:`m`.
        """
        return self._set(m=value)

    def getEf(self):
        """
        Gets the value of ef or its default value.
        """
        return self.getOrDefault(self.ef)

    def setEf(self, value):
        """
        Sets the value of :py:attr:`ef`.
        """
        return self._set(ef=value)

    def getEfConstruction(self):
        """
        Gets the value of efConstruction or its default value.
        """
        return self.getOrDefault(self.efConstruction)

    def setEfConstruction(self, value):
        """
        Sets the value of :py:attr:`efConstruction`.
        """
        return self._set(efConstruction=value)

    @keyword_only
    def setParams(self, identifierCol="id", vectorCol="vector", neighborsCol="neighbors",
                  m=16, ef=10, efConstruction=200, numPartitions=1, k=5, distanceFunction="cosine", excludeSelf=False,
                  similarityThreshold=-1.0, outputFormat="full", storageLevel="MEMORY_ONLY"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return HnswModel(java_model)


class HnswModel(JavaModel, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by Hnsw.
    """

    def __init__(self, java_model):
        super(HnswModel, self).__init__(java_model)

        # note: look at https://issues.apache.org/jira/browse/SPARK-10931 in the future

        self.identifierCol = Param(self, "identifierCol", "the column name for the row identifier")
        self.vectorCol = Param(self, "vectorCol", "the column name for the vector")
        self.neighborsCol = Param(self, "neighborsCol", "column names for returned neighbors")
        self.k = Param(self, "k", "number of neighbors to find")
        self.ef = Param(self, "ef", "size of the dynamic list for the nearest neighbors (used during the search)")
        self.excludeSelf = Param(self, "excludeSelf", "whether to include the row identifier as a candidate neighbor")
        self.similarityThreshold = Param(self, "similarityThreshold",
                                         "do not return neighbors further away than this distance")
        self.outputFormat = Param(self, "outputFormat", "output format, one of full, minimal")

        self._transfer_params_from_java()