package org.github.jelmerk.knn.hnsw;

import java.io.Serializable;

public interface LevelAssigner<TId> extends Serializable {

    int allocate(TId identifier);
}
