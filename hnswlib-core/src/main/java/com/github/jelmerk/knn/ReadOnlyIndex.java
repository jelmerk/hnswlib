package com.github.jelmerk.knn;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Read only K-nearest neighbours search index.
 *
 * @param <TId> type of the external identifier of an item
 * @param <TVector> The type of the vector to perform distance calculation on
 * @param <TItem> The type of items to connect into small world.
 * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
 *
 * @see <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbors algorithm</a>
 */
public interface ReadOnlyIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance> extends Serializable {

    /**
     * Returns the size of the index.
     *
     * @return size of the index
     */
    int size();

    /**
     * Returns an item by its identifier.
     *
     * @param id unique identifier or the item to return
     * @return an item
     */
    Optional<TItem> get(TId id);

    /**
     * Find the items closest to the passed in vector.
     *
     * @param vector the vector
     * @param k number of items to return
     * @return the items closest to the passed in vector
     */
    List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k);

    /**
     * Find the items closest to the item identified by the passed in id. If the id does not match an item an empty
     * list is returned. the element itself is not included in the response.
     *
     * @param id id of the item to find the neighbours of
     * @param k number of items to return
     * @return the items closest to the item
     */
    default List<SearchResult<TItem, TDistance>> findNeighbours(TId id, int k) {
        return get(id).map(item -> findNearest(item.vector(), k + 1).stream()
                .filter(result -> !result.item().id().equals(id))
                .limit(k)
                .collect(Collectors.toList()))
                .orElse(Collections.emptyList());
    }

    /**
     * Saves the index to an OutputStream.
     *
     * @param out the output stream to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    /**
     * Saves the index to a file.
     *
     * @param file file to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }

    /**
     * Saves the index to a path.
     *
     * @param path file to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(Path path) throws IOException {
        save(Files.newOutputStream(path));
    }
}
