[![Build Status](https://github.com/jelmerk/hnswlib/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jelmerk/hnswlib/actions/workflows/ci.yml)

Hnswlib
=======


Java implementation of the [the Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (HNSW) algorithm for doing approximate nearest neighbour search.

The index is thread safe, serializable, supports adding items to the index incrementally and has experimental support for deletes. 

It's flexible interface makes it easy to apply it to use it with any type of data and distance metric.
 
The following distance metrics are currently pre-packaged :

- bray curtis dissimilarity
- canberra distance
- correlation distance
- cosine distance
- euclidean distance
- inner product
- manhattan distance

It comes with a [scala wrapper](https://github.com/jelmerk/hnswlib/tree/master/hnswlib-scala)  that should feel native to scala developers

Apache spark support was moved into the [hnswlib-spark](https://github.com/jelmerk/hnswlib-spark) project.

To find out more about how to use this library take a look at the [hnswlib-examples](https://github.com/jelmerk/hnswlib/tree/master/hnswlib-examples) module or browse the documentation
in the readme files of the submodules

Sponsors
--------

![YourKIT logo](https://www.yourkit.com/images/yklogo.png)

YourKit is the creator of [YourKit Java Profiler](https://www.yourkit.com/java/profiler),
[YourKit .NET Profiler](https://www.yourkit.com/.net/profiler/),
and [YourKit YouMonitor](https://www.yourkit.com/youmonitor/).