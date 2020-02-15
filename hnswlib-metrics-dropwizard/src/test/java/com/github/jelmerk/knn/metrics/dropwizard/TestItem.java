package com.github.jelmerk.knn.metrics.dropwizard;

import com.github.jelmerk.knn.Item;

final class TestItem implements Item<String, float[]> {

    private static final long serialVersionUID = 1L;

    private final String id;
    private final float[] vector;

    TestItem(String id, float[] vector) {
        this.id = id;
        this.vector = vector;
    }

    @Override
    public int dimensions() {
        return vector.length;
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
