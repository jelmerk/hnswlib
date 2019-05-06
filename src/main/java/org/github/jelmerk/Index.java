package org.github.jelmerk;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

public interface Index<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>> {

    TItem get(TId id);

    void add(TItem item);

    List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k);

    void save(OutputStream out) throws IOException;

    default void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }
}