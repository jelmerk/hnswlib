from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable

__all__ = ['HnswSimilarity', 'HnswSimilarityModel', 'BruteForceSimilarity', 'BruteForceSimilarityModel']

@inherit_doc
class _KnnModelParams(HasFeaturesCol, HasPredictionCol):
    """
    Params for knn models.
    """

    queryIdentifierCol = Param(Params._dummy(), "queryIdentifierCol", "the column name for the query identifier",
                               typeConverter=TypeConverters.toString)

    k = Param(Params._dummy(), "k", "number of neighbors to find", typeConverter=TypeConverters.toInt)

    numReplicas = Param(Params._dummy(), "numReplicas", "number of index replicas to create when querying", typeConverter=TypeConverters.toInt)

    excludeSelf = Param(Params._dummy(), "excludeSelf", "whether to include the row identifier as a candidate neighbor",
                        typeConverter=TypeConverters.toBoolean)

    similarityThreshold = Param(Params._dummy(), "similarityThreshold",
                                "do not return neighbors further away than this distance",
                                typeConverter=TypeConverters.toFloat)

    outputFormat = Param(Params._dummy(), "outputFormat", "output format, one of full, minimal",
                         typeConverter=TypeConverters.toString)

    def getQueryIdentifierCol(self):
        """
        Gets the value of queryIdentifierCol or its default value.
        """
        return self.getOrDefault(self.queryIdentifierCol)

    def getK(self):
        """
        Gets the value of k or its default value.
        """
        return self.getOrDefault(self.k)

    def getExcludeSelf(self):
        """
        Gets the value of excludeSelf or its default value.
        """
        return self.getOrDefault(self.excludeSelf)

    def getSimilarityThreshold(self):
        """
        Gets the value of similarityThreshold or its default value.
        """
        return self.getOrDefault(self.similarityThreshold)

    def getOutputFormat(self):
        """
        Gets the value of outputFormat or its default value.
        """
        return self.getOrDefault(self.outputFormat)

    def getNumReplicas(self):
        """
        Gets the value of numReplicas or its default value.
        """
        return self.getOrDefault(self.numReplicas)

@inherit_doc
class _KnnParams(_KnnModelParams):
    """
    Params for knn algorithms.
    """

    identifierCol = Param(Params._dummy(), "identifierCol", "the column name for the row identifier",
                          typeConverter=TypeConverters.toString)

    numPartitions = Param(Params._dummy(), "numPartitions", "number of partitions", typeConverter=TypeConverters.toInt)

    distanceFunction = Param(Params._dummy(), "distanceFunction",
                             "distance function, one of bray-curtis, canberra, cosine, correlation, " +
                             "euclidean, inner-product, manhattan or the fully qualified classname " +
                             "of a distance function", typeConverter=TypeConverters.toString)

    def getIdentifierCol(self):
        """
        Gets the value of identifierCol or its default value.
        """
        return self.getOrDefault(self.identifierCol)

    def getNumPartitions(self):
        """
        Gets the value of numPartitions or its default value.
        """
        return self.getOrDefault(self.numPartitions)

    def getDistanceFunction(self):
        """
        Gets the value of distanceFunction or its default value.
        """
        return self.getOrDefault(self.distanceFunction)


@inherit_doc
class _HnswModelParams(_KnnModelParams):
    """
    Params for :py:class:`Hnsw` and :py:class:`HnswModel`.
    """

    ef = Param(Params._dummy(), "ef", "size of the dynamic list for the nearest neighbors (used during the search)",
               typeConverter=TypeConverters.toInt)

    def getEf(self):
        """
        Gets the value of ef or its default value.
        """
        return self.getOrDefault(self.ef)


@inherit_doc
class _HnswParams(_HnswModelParams, _KnnParams):
    """
    Params for :py:class:`Hnsw`.
    """

    m = Param(Params._dummy(), "m", "number of bi-directional links created for every new element during construction",
              typeConverter=TypeConverters.toInt)

    efConstruction = Param(Params._dummy(), "efConstruction",
                           "has the same meaning as ef, but controls the index time / index precision",
                           typeConverter=TypeConverters.toInt)

    def getM(self):
        """
        Gets the value of m or its default value.
        """
        return self.getOrDefault(self.m)

    def getEfConstruction(self):
        """
        Gets the value of efConstruction or its default value.
        """
        return self.getOrDefault(self.efConstruction)


@inherit_doc
class BruteForceSimilarity(JavaEstimator, _KnnParams, JavaMLReadable, JavaMLWritable):
    """
    Exact nearest neighbour search.
    """

    @keyword_only
    def __init__(self, identifierCol="id", queryIdentifierCol=None, featuresCol="features", predictionCol="prediction",
                 numPartitions=1, numReplicas=0, k=5, distanceFunction="cosine", excludeSelf=False,
                 similarityThreshold=-1.0, outputFormat="full"):
        super(BruteForceSimilarity, self).__init__()
        self._java_obj = self._new_java_obj("com.github.jelmerk.spark.knn.bruteforce.BruteForceSimilarity", self.uid)

        self._setDefault(identifierCol="id", numPartitions=1, numReplicas=0, k=5, distanceFunction="cosine",
                         excludeSelf=False, similarityThreshold=-1.0, outputFormat="full")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setIdentifierCol(self, value):
        """
        Sets the value of :py:attr:`identifierCol`.
        """
        return self._set(identifierCol=value)

    def setQueryIdentifierCol(self, value):
        """
        Sets the value of :py:attr:`queryIdentifierCol`.
        """
        return self._set(queryIdentifierCol=value)

    def setNumPartitions(self, value):
        """
        Sets the value of :py:attr:`numPartitions`.
        """
        return self._set(numPartitions=value)

    def setNumReplicas(self, value):
        """
        Sets the value of :py:attr:`numReplicas`.
        """
        return self._set(numReplicas=value)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set(k=value)

    def setDistanceFunction(self, value):
        """
        Sets the value of :py:attr:`distanceFunction`.
        """
        return self._set(distanceFunction=value)

    def setExcludeSelf(self, value):
        """
        Sets the value of :py:attr:`excludeSelf`.
        """
        return self._set(excludeSelf=value)

    def setSimilarityThreshold(self, value):
        """
        Sets the value of :py:attr:`similarityThreshold`.
        """
        return self._set(similarityThreshold=value)

    def setOutputFormat(self, value):
        """
        Sets the value of :py:attr:`outputFormat`.
        """
        return self._set(outputFormat=value)

    @keyword_only
    def setParams(self, identifierCol="id", queryIdentifierCol=None, featuresCol="features", predictionCol="prediction",
                  numPartitions=1, numReplicas=0, k=5, distanceFunction="cosine", excludeSelf=False,
                  similarityThreshold=-1.0, outputFormat="full"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return BruteForceSimilarityModel(java_model)


class BruteForceSimilarityModel(JavaModel, _KnnModelParams, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by BruteForce.
    """

    def setQueryIdentifierCol(self, value):
        """
        Sets the value of :py:attr:`queryIdentifierCol`.
        """
        return self._set(queryIdentifierCol=value)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set(k=value)

    def setExcludeSelf(self, value):
        """
        Sets the value of :py:attr:`excludeSelf`.
        """
        return self._set(excludeSelf=value)

    def setSimilarityThreshold(self, value):
        """
        Sets the value of :py:attr:`similarityThreshold`.
        """
        return self._set(similarityThreshold=value)

    def setOutputFormat(self, value):
        """
        Sets the value of :py:attr:`outputFormat`.
        """
        return self._set(outputFormat=value)

    def setNumReplicas(self, value):
        """
        Sets the value of :py:attr:`numReplicas`.
        """
        return self._set(numReplicas=value)


@inherit_doc
class HnswSimilarity(JavaEstimator, _HnswParams, JavaMLReadable, JavaMLWritable):
    """
    Approximate nearest neighbour search.
    """

    @keyword_only
    def __init__(self, identifierCol="id", queryIdentifierCol=None, featuresCol="features", predictionCol="prediction", m=16, ef=10,
                 efConstruction=200, numPartitions=1, numReplicas=0, k=5, distanceFunction="cosine", excludeSelf=False,
                 similarityThreshold=-1.0, outputFormat="full"):
        super(HnswSimilarity, self).__init__()
        self._java_obj = self._new_java_obj("com.github.jelmerk.spark.knn.hnsw.HnswSimilarity", self.uid)

        self._setDefault(identifierCol="id", m=16, ef=10, efConstruction=200, numPartitions=1, numReplicas=0, k=5,
                         distanceFunction="cosine", excludeSelf=False, similarityThreshold=-1.0, outputFormat="full")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setIdentifierCol(self, value):
        """
        Sets the value of :py:attr:`identifierCol`.
        """
        return self._set(identifierCol=value)

    def setQueryIdentifierCol(self, value):
        """
        Sets the value of :py:attr:`queryIdentifierCol`.
        """
        return self._set(queryIdentifierCol=value)

    def setNumPartitions(self, value):
        """
        Sets the value of :py:attr:`numPartitions`.
        """
        return self._set(numPartitions=value)

    def setNumReplicas(self, value):
        """
        Sets the value of :py:attr:`numReplicas`.
        """
        return self._set(numReplicas=value)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set(k=value)

    def setDistanceFunction(self, value):
        """
        Sets the value of :py:attr:`distanceFunction`.
        """
        return self._set(distanceFunction=value)

    def setExcludeSelf(self, value):
        """
        Sets the value of :py:attr:`excludeSelf`.
        """
        return self._set(excludeSelf=value)

    def setSimilarityThreshold(self, value):
        """
        Sets the value of :py:attr:`similarityThreshold`.
        """
        return self._set(similarityThreshold=value)

    def setOutputFormat(self, value):
        """
        Sets the value of :py:attr:`outputFormat`.
        """
        return self._set(outputFormat=value)

    def setM(self, value):
        """
        Sets the value of :py:attr:`m`.
        """
        return self._set(m=value)

    def setEf(self, value):
        """
        Sets the value of :py:attr:`ef`.
        """
        return self._set(ef=value)

    def setEfConstruction(self, value):
        """
        Sets the value of :py:attr:`efConstruction`.
        """
        return self._set(efConstruction=value)

    @keyword_only
    def setParams(self, identifierCol="id", queryIdentifierCol=None, featuresCol="features", predictionCol="prediction", m=16, ef=10,
                  efConstruction=200, numPartitions=1, numReplicas=0, k=5, distanceFunction="cosine", excludeSelf=False,
                  similarityThreshold=-1.0, outputFormat="full"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return HnswSimilarityModel(java_model)


class HnswSimilarityModel(JavaModel, _HnswModelParams, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by Hnsw.
    """

    def setQueryIdentifierCol(self, value):
        """
        Sets the value of :py:attr:`queryIdentifierCol`.
        """
        return self._set(queryIdentifierCol=value)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set(k=value)

    def setEf(self, value):
        """
        Sets the value of :py:attr:`ef`.
        """
        return self._set(ef=value)

    def setExcludeSelf(self, value):
        """
        Sets the value of :py:attr:`excludeSelf`.
        """
        return self._set(excludeSelf=value)

    def setSimilarityThreshold(self, value):
        """
        Sets the value of :py:attr:`similarityThreshold`.
        """
        return self._set(similarityThreshold=value)

    def setOutputFormat(self, value):
        """
        Sets the value of :py:attr:`outputFormat`.
        """
        return self._set(outputFormat=value)

    def setNumReplicas(self, value):
        """
        Sets the value of :py:attr:`numReplicas`.
        """
        return self._set(numReplicas=value)