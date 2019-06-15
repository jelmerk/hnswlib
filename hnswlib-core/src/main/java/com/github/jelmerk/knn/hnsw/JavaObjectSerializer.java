package com.github.jelmerk.knn.hnsw;

import java.io.*;

public class JavaObjectSerializer<T> implements ObjectSerializer<T> {

    private static final long serialVersionUID = 1L;

    @Override
    public void write(T item, ObjectOutput out) throws IOException {
        out.writeObject(item);
    }

    @Override
    @SuppressWarnings("unchecked")
    public T read(ObjectInput in) throws IOException, ClassNotFoundException {
        return (T) in.readObject();
    }

}
