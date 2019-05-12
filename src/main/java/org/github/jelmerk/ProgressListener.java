package org.github.jelmerk;

@FunctionalInterface
public interface ProgressListener {

    void updateProgress(int workDone, int max);
}
