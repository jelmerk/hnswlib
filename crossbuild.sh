#!/usr/bin/env bash

set -e

function cleanup()
{
    git checkout ./hnswlib-scala/pom.xml
    git checkout ./hnswlib-spark/pom.xml
}

trap cleanup EXIT

ARGS=$*

mvn $ARGS -N

mvn $ARGS -pl hnswlib-utils

mvn $ARGS -pl hnswlib-core

mvn $ARGS -pl hnswlib-metrics-dropwizard

mvn $ARGS -pl hnswlib-core-jdk17

cp ./hnswlib-scala/pom-scala-2_11.xml ./hnswlib-scala/pom.xml
mvn $ARGS -pl hnswlib-scala

cp ./hnswlib-scala/pom-scala-2_12.xml ./hnswlib-scala/pom.xml
mvn $ARGS -pl hnswlib-scala

cp ./hnswlib-scala/pom-scala-2_13.xml ./hnswlib-scala/pom.xml
mvn $ARGS -pl hnswlib-scala

cp ./hnswlib-spark/pom-spark-2.3-scala-2_11.xml ./hnswlib-spark/pom.xml
mvn $ARGS -pl hnswlib-spark

cp ./hnswlib-spark/pom-spark-2.4-scala-2_11.xml ./hnswlib-spark/pom.xml
mvn $ARGS -pl hnswlib-spark

cp ./hnswlib-spark/pom-spark-2.4-scala-2_12.xml ./hnswlib-spark/pom.xml
mvn $ARGS -pl hnswlib-spark

cp ./hnswlib-spark/pom-spark-3.0-scala-2_12.xml ./hnswlib-spark/pom.xml
mvn $ARGS -pl hnswlib-spark

cp ./hnswlib-spark/pom-spark-3.1-scala-2_12.xml ./hnswlib-spark/pom.xml
mvn $ARGS -pl hnswlib-spark

cp ./hnswlib-spark/pom-spark-3.2-scala-2_12.xml ./hnswlib-spark/pom.xml
mvn $ARGS -pl hnswlib-spark

cp ./hnswlib-spark/pom-spark-3.3-scala-2_12.xml ./hnswlib-spark/pom.xml
mvn $ARGS -pl hnswlib-spark


mvn $ARGS -pl hnswlib-examples