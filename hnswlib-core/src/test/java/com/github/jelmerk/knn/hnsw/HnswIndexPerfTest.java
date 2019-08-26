package com.github.jelmerk.knn.hnsw;


import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.Item;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HnswIndexPerfTest {

    static class MyItem implements Item<Integer, float[]> { //}, Externalizable {
        private Integer id;
        private float[] vector;

        public MyItem() {
        }

        MyItem(Integer id, float[] vector) {
            this.id = id;
            this.vector = vector;
        }

        @Override
        public Integer id() {
            return id;
        }

        @Override
        public float[] vector() {
            return vector;
        }


//        @Override
        public void writeExternal(ObjectOutput out) throws IOException {
            out.writeInt(id);
            out.writeInt(vector.length);

            for (int i = 0; i < vector.length; i++) {
                out.writeFloat(vector[i]);
            }
        }

//        @Override
        public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {

        }
    }

    static class IdSerializer implements ObjectSerializer<Integer> {

        @Override
        public void write(Integer item, ObjectOutput out) throws IOException {
            out.writeInt(item);
        }

        @Override
        public Integer read(ObjectInput in) throws IOException, ClassNotFoundException {
            return null;
        }
    }


    static class ItemSerializer implements ObjectSerializer<MyItem> {

        @Override
        public void write(MyItem item, ObjectOutput out) throws IOException {

            out.writeInt(item.id);
            out.writeInt(item.vector.length);

            for (int i = 0; i < item.vector.length; i++) {
                out.writeFloat(item.vector[i]);
            }
        }

        @Override
        public MyItem read(ObjectInput in) throws IOException, ClassNotFoundException {
            return null;
        }
    }


    private static final Random random = new Random(42);

    public static void main(String[] args) throws Exception {


        List<MyItem> items = generateRandomItems(200_000, 300);

        System.out.println("Done generating random vectors.");

        long start = System.currentTimeMillis();

        int m = 10;

        HnswIndex<Integer, float[], MyItem, Float> index = HnswIndex
                .newBuilder(DistanceFunctions.FLOAT_INNER_PRODUCT, items.size())
                .withCustomSerializers(new IdSerializer(), new ItemSerializer())
                .withM(m)
                .build();

//        for (MyItem item : items) {
//            index.add(item);
//        }

        index.addAll(items);

        long end = System.currentTimeMillis();


        long duration = end - start;

        System.out.println("Done creating index. took : " + duration + "ms");

        while(true) {
            long startSave = System.currentTimeMillis();
            index.save(new ByteArrayOutputStream());

            long endSave = System.currentTimeMillis();

            long saveDuration = endSave - startSave;

            System.out.println("save took " + saveDuration);
        }
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
