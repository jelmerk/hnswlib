package org.github.jelmerk.hnsw;

import org.github.jelmerk.Item;

import java.io.Serializable;

public class TestItem implements Item<String, float[]>, Serializable {

    private final String id;
    private float[] vector;

    public TestItem(String id, float[] vector) {
        this.id = id;
        this.vector = vector;
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public float[] getVector() {
        return vector;
    }

    public void setVector(float[] vector) {
        this.vector = vector;
    }
}