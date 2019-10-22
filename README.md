[![Build Status](https://travis-ci.org/jelmerk/hnswlib.svg?branch=master)](https://travis-ci.org/jelmerk/hnswlib)

Hnswlib
=======


Work in progress java implementation of the [the Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (HNSW) algorithm for doing approximate nearest neighbour search.

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

It comes with [apache spark](https://spark.apache.org/) integration and a [scala](https://scala-lang.org) wrapper that should feel native to scala developers 

To find out more about how to use this library take a look at the [hnswlib-examples](https://github.com/jelmerk/hnswlib/tree/master/hnswlib-examples) module or browse the documentation
in the readme files of the submodules