package org.github.jelmerk;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

public interface NearestNeighboursAlgorithm<TItem, TDistance extends Comparable<TDistance>> {

    TItem getItemById(int id);

    int addItem(TItem item);

    List<SearchResult<TItem, TDistance>> search(TItem item, int k);

    void saveIndex(OutputStream out) throws IOException;

    default void saveIndex(File file) throws IOException {
        saveIndex(new FileOutputStream(file));
    }
}