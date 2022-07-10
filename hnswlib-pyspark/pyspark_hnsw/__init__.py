import subprocess
import threading
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.java_gateway import launch_gateway


def start(spark23=False,
          spark24=False,
          spark31=False,
          memory="16G",
          cache_folder="/tmp",
          real_time_output=False,
          output_level=1):
    """Starts a PySpark instance with default parameters for Hnswlib.

    The default parameters would result in the equivalent of:

    .. code-block:: python
        :param spark23: start Hnswlib on Apache Spark 2.3.x
        :param spark24: start Hnswlib on Apache Spark 2.4.x
        :param spark31: start Hnswlib on Apache Spark 3.1.x
        :param memory: set driver memory for SparkSession
        :param output_level: int, optional Output level for logs, by default 1
        :param real_time_output:
        :substitutions:

        SparkSession.builder \\
            .appName("Hnswlib") \\
            .master("local[*]") \\
            .config("spark.driver.memory", "16G") \\
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \\
            .config("spark.kryo.registrator", "com.github.jelmerk.spark.HnswLibKryoRegistrator") \\
            .config("spark.jars.packages", "com.github.jelmerk:hnswlib-spark_3.0_2.12:|release|") \\
            .config("spark.hnswlib.settings.index.cache_folder", "/tmp") \\
            .getOrCreate()


    Parameters
    ----------

    spark23 : bool, optional
        Whether to use the Spark 2.3.x version of Hnswlib, by default False
    spark24 : bool, optional
        Whether to use the Spark 2.4.x version of Hnswlib, by default False
    spark31 : bool, optional
        Whether to use the Spark 3.1.x version of Hnswlib, by default False
    memory : str, optional
        How much memory to allocate for the Spark driver, by default "16G"
    real_time_output : bool, optional
        Whether to output in real time, by default False
    output_level : int, optional
        Output level for logs, by default 1

    Returns
    -------
    :class:`SparkSession`
        The initiated Spark session.

    """
    current_version = "1.1.0"

    class HnswlibConfig:

        def __init__(self):
            self.master = "local[*]"
            self.app_name = "Hnswlib"
            self.serializer = "org.apache.spark.serializer.KryoSerializer"
            self.registrator = "com.github.jelmerk.spark.HnswLibKryoRegistrator"
            # Hnswlib on Apache Spark 3.2.x

            # Hnswlib on Apache Spark 3.0.x/3.1.x
            self.maven_spark = "com.github.jelmerk:hnswlib-spark_3.1_2.12:{}".format(current_version)
            # Hnswlib on Apache Spark 2.4.x
            self.maven_spark24 = "com.github.jelmerk:hnswlib-spark_2.4_2.12:{}".format(current_version)
            # Hnswlib on Apache Spark 2.3.x
            self.maven_spark23 = "com.github.jelmerk:hnswlib-spark_2.3_2.11:{}".format(current_version)

    def start_without_realtime_output():
        builder = SparkSession.builder \
            .appName(spark_nlp_config.app_name) \
            .master(spark_nlp_config.master) \
            .config("spark.driver.memory", memory) \
            .config("spark.serializer", spark_nlp_config.serializer) \
            .config("spark.kryo.registrator", spark_nlp_config.registrator) \
            .config("spark.hnswlib.settings.index.cache_folder", cache_folder)

        if spark23:
            builder.config("spark.jars.packages", spark_nlp_config.maven_spark23)
        elif spark24:
            builder.config("spark.jars.packages", spark_nlp_config.maven_spark24)
        else:
            builder.config("spark.jars.packages", spark_nlp_config.maven_spark)

        return builder.getOrCreate()

    def start_with_realtime_output():

        class SparkWithCustomGateway:

            def __init__(self):
                spark_conf = SparkConf()
                spark_conf.setAppName(spark_nlp_config.app_name)
                spark_conf.setMaster(spark_nlp_config.master)
                spark_conf.set("spark.driver.memory", memory)
                spark_conf.set("spark.serializer", spark_nlp_config.serializer)
                spark_conf.set("spark.kryo.registrator", spark_nlp_config.registrator)
                spark_conf.set("spark.jars.packages", spark_nlp_config.maven_spark)
                spark_conf.set("spark.hnswlib.settings.index.cache_folder", cache_folder)

                # Make the py4j JVM stdout and stderr available without buffering
                popen_kwargs = {
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.PIPE,
                    'bufsize': 0
                }

                # Launch the gateway with our custom settings
                self.gateway = launch_gateway(conf=spark_conf, popen_kwargs=popen_kwargs)
                self.process = self.gateway.proc
                # Use the gateway we launched
                spark_context = SparkContext(gateway=self.gateway)
                self.spark_session = SparkSession(spark_context)

                self.out_thread = threading.Thread(target=self.output_reader)
                self.error_thread = threading.Thread(target=self.error_reader)
                self.std_background_listeners()

            def std_background_listeners(self):
                self.out_thread.start()
                self.error_thread.start()

            def output_reader(self):
                for line in iter(self.process.stdout.readline, b''):
                    print('{0}'.format(line.decode('utf-8')), end='')

            def error_reader(self):
                RED = '\033[91m'
                RESET = '\033[0m'
                for line in iter(self.process.stderr.readline, b''):
                    if output_level == 0:
                        print(RED + '{0}'.format(line.decode('utf-8')) + RESET, end='')
                    else:
                        # output just info
                        pass

            def shutdown(self):
                self.spark_session.stop()
                self.gateway.shutdown()
                self.process.communicate()

                self.out_thread.join()
                self.error_thread.join()

        return SparkWithCustomGateway()

    spark_nlp_config = HnswlibConfig()

    if real_time_output:
        if spark23 or spark24:
            spark_session = start_without_realtime_output()
            return spark_session
        else:
            # Available from Spark 3.0.x
            class SparkRealTimeOutput:

                def __init__(self):
                    self.__spark_with_custom_gateway = start_with_realtime_output()
                    self.spark_session = self.__spark_with_custom_gateway.spark_session

                def shutdown(self):
                    self.__spark_with_custom_gateway.shutdown()

            return SparkRealTimeOutput()
    else:
        spark_session = start_without_realtime_output()
        return spark_session


def version():
    """Returns the current Hnswlib version.

    Returns
    -------
    str
        The current Hnswlib version.
    """
    return '1.1.0'
