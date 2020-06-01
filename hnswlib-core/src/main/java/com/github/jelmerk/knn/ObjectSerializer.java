package com.github.jelmerk.knn;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serializable;

/**
 * Implementations of this interface are used to customize how objects will be stored when the index is persisted
 *
 * @param <T> type of object to serialize.
 */
public interface ObjectSerializer<T> extends Serializable {

    /**
     * Writes the item to an ObjectOutput implementation.
     *
     * @param item the item to write
     * @param out the ObjectOutput implementation to write to
     * @throws IOException in case of an I/O exception
     */
    void write(T item, ObjectOutput out) throws IOException;

    /**
     * Reads an item from an ObjectOutput implementation.
     *
     * @param in the ObjectInput implementation to read from
     * @return the read item
     * @throws IOException in case of an I/O exception
     * @throws ClassNotFoundException in case the value read does not match the type of item
     */
    T read(ObjectInput in) throws IOException, ClassNotFoundException;

}
