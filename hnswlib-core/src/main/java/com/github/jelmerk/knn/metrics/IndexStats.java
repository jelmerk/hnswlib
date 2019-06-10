package com.github.jelmerk.knn.metrics;

import java.io.Serializable;
import java.util.Objects;

public class IndexStats implements Serializable {

    private static final long serialVersionUID = 1L;

    private final double accuracy;

    public IndexStats(double accuracy) {
        this.accuracy = accuracy;
    }

    public double accuracy() {
        return accuracy;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IndexStats that = (IndexStats) o;
        return Double.compare(that.accuracy, accuracy) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(accuracy);
    }

    @Override
    public String toString() {
        return "IndexStats{" +
                "accuracy=" + accuracy +
                '}';
    }
}
