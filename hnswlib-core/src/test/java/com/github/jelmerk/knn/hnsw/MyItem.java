package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.Item;

public class MyItem implements Item<Integer, float[]> {

    private final int id;
    private final float[] vector;

    public MyItem(int id, float[] vector) {
        this.id = id;
        this.vector = vector;
    }

    @Override
    public Integer id() {
        return id;
    }

    @Override
    public float[] vector() {
        return vector;
    }
}
