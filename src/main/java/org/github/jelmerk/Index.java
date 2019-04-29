package org.github.jelmerk;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

public interface Index<TItem, TDistance extends Comparable<TDistance>> {

    int add(TItem item);

    List<SearchResult<TItem, TDistance>> search(TItem item, int k);

    void save(OutputStream out) throws IOException;

    default void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }
}