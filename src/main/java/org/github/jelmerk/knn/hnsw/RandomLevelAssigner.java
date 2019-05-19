package org.github.jelmerk.knn.hnsw;

import java.util.Random;

public class RandomLevelAssigner<TId> implements LevelAssigner<TId> {

    private final Random generator;
    private final double poissonLambda;

    public RandomLevelAssigner(long seed, double poissonLambda) {
        this.generator = new Random(seed);
        this.poissonLambda = poissonLambda;
    }

    public RandomLevelAssigner(double poissonLambda) {
        this(System.currentTimeMillis(), poissonLambda);
    }

    @Override
    public int allocate(TId identifier) {
        double random = generator.nextDouble();
        double r = -Math.log(random) * poissonLambda;
        return (int)r;
    }
}
