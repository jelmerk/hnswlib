package org.github.jelmerk.hnsw;

import java.io.Serializable;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.function.Supplier;

class Pool<T> implements Serializable {

    private final ArrayBlockingQueue<T> items;

    Pool(Supplier<T> supplier, int maxPoolSize) {
        this.items = new ArrayBlockingQueue<>(maxPoolSize);

        for (int i = 0; i < maxPoolSize; i++) {
            items.add(supplier.get());
        }
    }

    T borrowObject() {
        try {
            return items.take();
        } catch (InterruptedException e) {
            throw new RuntimeException(e); // TODO jk any more elegant way to do this ?
        }
    }

    void returnObject(T item) {
        items.add(item);
    }


}
