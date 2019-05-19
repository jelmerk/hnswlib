package org.github.jelmerk.knn.hnsw;

import static java.nio.charset.StandardCharsets.UTF_8;

public class StringMurmur3LevelAllocator extends AbstractMurmur3LevelAssigner<String> {

    public StringMurmur3LevelAllocator(double poissonLambda) {
        super(poissonLambda);
    }

    @Override
    protected byte[] toBytes(String identifier) {
        return identifier.getBytes(UTF_8);
    }
}
