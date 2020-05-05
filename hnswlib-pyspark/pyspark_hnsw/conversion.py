from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.wrapper import JavaTransformer
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.mllib.common import inherit_doc
from pyspark import keyword_only

__all__ = ['VectorConverter']

@inherit_doc
class VectorConverter(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable):
    """
    Converts the input vector to a float array.
    """

    @keyword_only
    def __init__(self, inputCol="input", outputCol="output"):
        """
        __init__(self, inputCol="input", outputCol="output")
        """
        super(VectorConverter, self).__init__()
        self._java_obj = self._new_java_obj("com.github.jelmerk.spark.conversion.VectorConverter", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol="input", outputCol="output"):
        """
        setParams(self, inputCol="input", outputCol="output")
        Sets params for this VectorConverter.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)
