package org.github.jelmerk.knn.hnsw;

public abstract class AbstractMurmur3LevelAssigner<TId> implements LevelAssigner<TId> {

    private final double poissonLambda;

    public AbstractMurmur3LevelAssigner(double poissonLambda) {
        this.poissonLambda = poissonLambda;
    }

    protected abstract byte[] toBytes(TId identifier);

    @Override
    public final int allocate(TId identifier) {
        double random = Math.abs((double) Murmur3.hash32(toBytes(identifier)) / (double) Integer.MAX_VALUE);
        double r = -Math.log(random) * poissonLambda;
        return (int)r;
    }
}
