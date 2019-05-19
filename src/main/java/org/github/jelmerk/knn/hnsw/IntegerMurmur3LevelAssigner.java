package org.github.jelmerk.knn.hnsw;

public class IntegerMurmur3LevelAssigner extends AbstractMurmur3LevelAssigner<Integer> {

    public IntegerMurmur3LevelAssigner(double poissonLambda) {
        super(poissonLambda);
    }

    @Override
    protected byte[] toBytes(Integer identifier) {
        return new byte[] {
                (byte) (identifier >> 24),
                (byte) (identifier >> 16),
                (byte) (identifier >> 8),
                (byte) (int) identifier
        };
    }
}
