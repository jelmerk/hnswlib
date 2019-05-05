package org.github.jelmerk.foo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.PriorityQueue;

public interface AlgorithmInterface<ID, ITEM extends Item<ID>> {

    void addPoint(ITEM item);

    ITEM getById(ID id);

    PriorityQueue<SearchResult<ITEM>> searchKnn(float[] vector, int k); // the first element of the pair is the distance, the second the label

    void saveIndex(OutputStream out) throws IOException;

    default void saveIndex(File file) throws IOException {
        saveIndex(new FileOutputStream(file));
    }
}


