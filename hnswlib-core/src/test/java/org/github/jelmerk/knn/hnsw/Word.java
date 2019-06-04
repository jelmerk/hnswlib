package org.github.jelmerk.knn.hnsw;

import org.github.jelmerk.knn.Item;

import java.util.Arrays;
import java.util.Objects;

public class Word implements Item<String, float[]> {

    private static final long serialVersionUID = 6845177627057649549L;

    private final String id;
    private final float[] vector;

    public Word(String id, float[] vector) {
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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Word word = (Word) o;
        return Objects.equals(id, word.id) &&
                Arrays.equals(vector, word.vector);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(id);
        result = 31 * result + Arrays.hashCode(vector);
        return result;
    }

    @Override
    public String toString() {
        return "Word{" +
                "id='" + id + '\'' +
                ", vector=" + Arrays.toString(vector) +
                '}';
    }
}