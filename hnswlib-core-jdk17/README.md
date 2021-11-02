[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-core-jdk17/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-core-jdk17) [![Javadoc](https://javadoc.io/badge2/com.github.jelmerk/hnswlib-core-jdk17/javadoc.svg)](https://javadoc.io/doc/com.github.jelmerk/hnswlib-core-jdk17)

hnswlib-core-jdk17
==================

Distance function implementations built on top of the [JEP 414 vector api](https://openjdk.java.net/jeps/414)

In most cases these implementations will be faster than their counterparts in `com.github.jelmerk.knn.DistanceFunctions`
even when auto-vectorization is happening

In order to make use of these functions you need to start your jvm with the following arguments

`--enable-preview --add-modules jdk.incubator.vector`