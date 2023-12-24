from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaTransformer
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only

__all__ = ['VectorConverter']

@inherit_doc
class VectorConverter(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable):
    """
    Converts the input vector to a vector of another type.
    """

    outputType = Param(Params._dummy(), "outputType", "type of vector to produce. one of array<float>, array<double>, vector",
                       typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, inputCol="input", outputCol="output", outputType="array<float>"):
        """
        __init__(self, inputCol="input", outputCol="output", outputType="array<float>")
        """
        super(VectorConverter, self).__init__()
        self._java_obj = self._new_java_obj("com.github.jelmerk.spark.conversion.VectorConverter", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol="input", outputCol="output", outputType="array<float>"):
        """
        setParams(self, inputCol="input", outputCol="output", outputType="array<float>")
        Sets params for this VectorConverter.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)
