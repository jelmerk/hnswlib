package com.github.jelmerk.knn;

import java.io.*;

/**
 * Implementation of {@link ObjectSerializer} that uses java serialization to write the value.
 *
 * @param <T> type of object to serialize
 */
public class JavaObjectSerializer<T> implements ObjectSerializer<T> {

    private static final long serialVersionUID = 1L;

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(T item, ObjectOutput out) throws IOException {
        out.writeObject(item);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    @SuppressWarnings("unchecked")
    public T read(ObjectInput in) throws IOException, ClassNotFoundException {
        return (T) in.readObject();
    }

}
