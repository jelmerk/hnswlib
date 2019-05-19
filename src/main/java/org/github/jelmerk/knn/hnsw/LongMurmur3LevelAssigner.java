package org.github.jelmerk.knn.hnsw;

public class LongMurmur3LevelAssigner extends AbstractMurmur3LevelAssigner<Long> {

    public LongMurmur3LevelAssigner(double poissonLambda) {
        super(poissonLambda);
    }

    @Override
    protected byte[] toBytes(Long identifier) {
        return new byte[] {
                (byte) (identifier >> 56),
                (byte) (identifier >> 48),
                (byte) (identifier >> 40),
                (byte) (identifier >> 32),
                (byte) (identifier >> 24),
                (byte) (identifier >> 16),
                (byte) (identifier >> 8),
                (byte) (long) identifier
        };
    }
}
