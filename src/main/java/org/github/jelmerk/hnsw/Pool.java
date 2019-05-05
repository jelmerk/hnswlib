package org.github.jelmerk.hnsw;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.function.Supplier;

public class Pool<T> {

    private final ArrayBlockingQueue<T> items;

    public Pool(Supplier<T> supplier , int maxPoolSize) {
        this.items = new ArrayBlockingQueue<>(maxPoolSize);

        for (int i = 0; i < maxPoolSize; i++) {
            items.add(supplier.get());
        }
    }

    public T borrowObject() {
        try {
            return items.take();
        } catch (InterruptedException e) {
            throw new RuntimeException(e); // TODO jk any more elegant way to do this ?
        }
    }

    public void returnObject(T item) {
        items.add(item);
    }


}
