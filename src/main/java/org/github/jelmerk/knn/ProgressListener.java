package org.github.jelmerk.knn;

@FunctionalInterface
public interface ProgressListener {

    void updateProgress(int workDone, int max);
}
