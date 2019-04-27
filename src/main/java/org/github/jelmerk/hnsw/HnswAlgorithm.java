package org.github.jelmerk.hnsw;


import org.github.jelmerk.NearestNeighboursAlgorithm;
import org.github.jelmerk.SearchResult;

import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

public class HnswAlgorithm<TItem, TDistance extends Comparable<TDistance>>
        implements NearestNeighboursAlgorithm<TItem, TDistance> {

    private final DotNetRandom random;
    private final Parameters parameters;
    private final DistanceFunction<TItem, TDistance> distanceFunction;

    public HnswAlgorithm(Parameters parameters,
                         DistanceFunction<TItem, TDistance> distanceFunction) {

        this(new DotNetRandom(), parameters, distanceFunction);
    }

    public HnswAlgorithm(DotNetRandom random,
                         Parameters parameters,
                         DistanceFunction<TItem, TDistance> distanceFunction) {

        this.random = random;
        this.parameters = parameters;
        this.distanceFunction = distanceFunction;
    }

    @Override
    public void addPoint(Object o) {

    }

    @Override
    public List<SearchResult<TItem, TDistance>> search(Object o, int k) {
        return null;
    }

    @Override
    public void saveIndex(OutputStream out) throws IOException {

    }
}
