package com.github.jelmerk.knn.hnsw;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serializable;

public interface ObjectSerializer<T> extends Serializable {

    void write(T item, ObjectOutput out) throws IOException;

    T read(ObjectInput in) throws IOException, ClassNotFoundException;

}
