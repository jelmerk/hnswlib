package org.github.jelmerk.knn.hnsw;


import org.github.jelmerk.knn.DistanceFunctions;
import org.github.jelmerk.knn.Item;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
//import jdk.incubator.vector.IntVetigctor;

public class HnswIndexPerfTest2 {

    static class MyItem implements Item<Integer, float[]> {
        private final Integer id;
        private final float[] vector;


        MyItem(Integer id, float[] vector) {
            this.id = id;
            this.vector = vector;
        }

        @Override
        public Integer getId() {
            return id;
        }

        @Override
        public float[] getVector() {
            return vector;
        }
    }

    private static final Random random = new Random(42);

    public static void main(String[] args) throws Exception {

        List<MyItem> items = generateRandomItems(2_000_000, 90);

        System.out.println("Done generating random vectors.");

        long start = System.currentTimeMillis();

        int m = 15;
        double poissonLambda = 1 / Math.log(m);

        HnswIndex<Integer, float[], MyItem, Float> index =
                new HnswIndex.Builder<>(DistanceFunctions::cosineDistance, new RandomLevelAssigner<Integer>(poissonLambda), items.size())
                        .setM(m)
                        .build();

//        for (MyItem item : items) {
//            index.add(item);
//        }

        index.addAll(items);

        long end = System.currentTimeMillis();


        long duration = end - start;

        System.out.println("Done creating small world. took : " + duration + "ms");

        index.save(new File("/Users/jkuperus/2_million_90_dimensions.ser"));
    }

    private static List<MyItem> generateRandomItems(int numItems, int vectorSize) {
        List<MyItem> result = new ArrayList<>(numItems);
        for (int i = 0; i < numItems; i++) {


            float[] vector = generateRandomVector(vectorSize);
            result.add(new MyItem(i, vector));
        }
        return result;
    }

    private static float[] generateRandomVector(int size) {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) {
            result[i] = random.nextFloat();
        }
        return result;
    }

}
