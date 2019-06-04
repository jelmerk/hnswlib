package org.github.jelmerk.knn.hnsw;

import org.github.jelmerk.knn.Item;

import java.io.Serializable;

public class TestItem implements Item<String, float[]>, Serializable {

    private final String id;
    private float[] vector;

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

    public void setVector(float[] vector) {
        this.vector = vector;
    }
}