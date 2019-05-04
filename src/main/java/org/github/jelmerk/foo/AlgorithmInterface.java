package org.github.jelmerk.foo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.PriorityQueue;

public interface AlgorithmInterface {

    void addPoint(float[] dataPoint);

    // TODO JK : do we want to explicitly expose a priority queue or just use a list here
    PriorityQueue<float[]> searchKnn(float[] item, int k);

    void saveIndex(OutputStream out) throws IOException;

    default void saveIndex(File file) throws IOException {
        saveIndex(new FileOutputStream(file));
    }
}


