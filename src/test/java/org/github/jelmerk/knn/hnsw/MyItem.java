package org.github.jelmerk.knn.hnsw;

import org.github.jelmerk.knn.Item;

public class MyItem implements Item<Integer, float[]> {

    private final int id;
    private final float[] vector;

    public MyItem(int id, float[] vector) {
        this.id = id;
        this.vector = vector;
    }

    @Override
    public Integer getId() {
        return id;
    }

    @Override
    public float[] getVector() {
        return vector;
    }
}
