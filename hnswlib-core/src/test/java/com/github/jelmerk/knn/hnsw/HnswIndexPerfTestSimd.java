package com.github.jelmerk.knn.hnsw;

//import jdk.incubator.vector.FloatVector;
//import jdk.incubator.vector.VectorSpecies;
//import com.github.jelmerk.Item;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HnswIndexPerfTestSimd {

//    static class MyItem implements Item<Integer, FloatVector> {
//        private final Integer id;
//        private final FloatVector vector;
//
//
//        MyItem(Integer id, FloatVector vector) {
//            this.id = id;
//            this.vector = vector;
//        }
//
//        @Override
//        public Integer id() {
//            return id;
//        }
//
//        @Override
//        public FloatVector vector() {
//            return vector;
//        }
//    }
//
//    private static final Random random = new Random(42);
//
//    public static void main(String[] args) throws Exception {
//
//
////        List<MyItem> items = generateRandomItems(100_000, 64);
////
////        System.out.println("Done generating random vectors.");
////
////        long start = System.currentTimeMillis();
////
////        int m = 15;
////
////        HnswIndex<Integer, FloatVector, MyItem, Float> index =
////                new HnswIndex.Builder<>(CosineDistanceSimd::simd, items.size())
////                        .withM(m)
////                        .setLevelLambda(1 / Math.log(m))
////                        .build();
////
////        index.addAll(items);
////
////        long end = System.currentTimeMillis();
////
////
////        long duration = end - start;
////
////        System.out.println("Done creating small world. took : " + duration + "ms");
////
//////        index.save(new File("/Users/jkuperus/graph.ser"));
//    }
//
//    private static List<MyItem> generateRandomItems(int numItems, int vectorSize) {
//        List<MyItem> result = new ArrayList<>(numItems);
//        for (int i = 0; i < numItems; i++) {
//
//
//
//
//            float[] vector = generateRandomVector(vectorSize);
//
//
//
//
//            FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector, 1);
//
////            result.add(new MyItem(i, vector));
//        }
//        return result;
//    }
//
//    private static float[] generateRandomVector(int size) {
//        float[] result = new float[size];
//        for (int i = 0; i < size; i++) {
//            result[i] = random.nextFloat();
//        }
//        return result;
//    }

}

