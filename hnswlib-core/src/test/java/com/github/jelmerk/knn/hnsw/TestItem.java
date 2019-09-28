package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.Item;

import java.io.Serializable;

public class TestItem implements Item<String, float[]> {

    private final String id;
    private final float[] vector;

    public TestItem(String id, float[] vector) {
        this.id = id;
        this.vector = vector;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public float[] vector() {
        return vector;
    }

}