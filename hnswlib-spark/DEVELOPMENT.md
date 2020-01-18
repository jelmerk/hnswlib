Development
-----------

The easiest way to test changes to the hnswlib codebase is to produce an assembly file with

    mvn clean assembly:assembly
    
And then reference it from spark with

    spark-submit --jars hnswlib-spark/target/hnswlib-spark-*-jar-with-dependencies.jar your.jar

