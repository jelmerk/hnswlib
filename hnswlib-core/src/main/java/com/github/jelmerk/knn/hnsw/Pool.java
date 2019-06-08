package com.github.jelmerk.knn.hnsw;

import java.io.Serializable;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.function.Supplier;

/**
 * Generic object pool.
 *
 * @param <T> type of object to pool
 */
class Pool<T> implements Serializable {

    private static final long serialVersionUID = 1L;

    private final ArrayBlockingQueue<T> items;

    /**
     * Constructs a new pool
     *
     * @param supplier used to create instances of the object to pool
     * @param maxPoolSize maximum items to have in the pool
     */
    Pool(Supplier<T> supplier, int maxPoolSize) {
        this.items = new ArrayBlockingQueue<>(maxPoolSize);

        for (int i = 0; i < maxPoolSize; i++) {
            items.add(supplier.get());
        }
    }

    /**
     * Borrows an object from the pool.
     *
     * @return the borrowed object
     */
    T borrowObject() {
        try {
            return items.take();
        } catch (InterruptedException e) {
            throw new RuntimeException(e); // TODO jk any more elegant way to do this ?
        }
    }

    /**
     * Returns an instance to the pool. By contract, obj must have been obtained using {@link Pool#borrowObject()}
     *
     * @param item the item to return to the pool
     */
    void returnObject(T item) {
        items.add(item);
    }

}
