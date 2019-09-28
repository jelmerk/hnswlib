package com.github.jelmerk.knn;

import java.util.Objects;

public class ProgressUpdate {

    private final int workDone;
    private final int max;

    public ProgressUpdate(int workDone, int max) {
        this.workDone = workDone;
        this.max = max;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ProgressUpdate that = (ProgressUpdate) o;
        return workDone == that.workDone &&
                max == that.max;
    }

    @Override
    public int hashCode() {
        return Objects.hash(workDone, max);
    }

    @Override
    public String toString() {
        return "ProgressUpdate{" +
                "workDone=" + workDone +
                ", max=" + max +
                '}';
    }

}