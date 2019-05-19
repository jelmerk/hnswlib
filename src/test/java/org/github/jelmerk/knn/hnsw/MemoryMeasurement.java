package org.github.jelmerk.knn.hnsw;

import org.github.jamm.MemoryMeter;
import org.github.jelmerk.knn.DistanceFunctions;
import org.github.jelmerk.knn.Index;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class MemoryMeasurement {


    // -javaagent:/Users/jkuperus/.m2/repository//com/github/jbellis/jamm/0.3.0/jamm-0.3.0.jar
    public static void main(String[] args) throws Exception {

        List<MyItem> allElements = loadItemsToIndex();

        int increment = 100_000;

        long lastSeenMemory = -1L;

        for (int i = increment; i <= allElements.size(); i += increment) {

            List<MyItem> items = allElements.subList(0, i);

            long memoryUsed = createIndexAndMeaureMemory(items);

            if (lastSeenMemory == -1) {
                System.out.printf("Memory used for index of size %d is %d bytes%n", i, memoryUsed);
            } else {
                System.out.printf("Memory used for index of size %d is %d bytes, delta with last generated index : %d bytes%n", i, memoryUsed, memoryUsed - lastSeenMemory);

            }

            lastSeenMemory = memoryUsed;

            createIndexAndMeaureMemory(items);

        }
    }

    private static long createIndexAndMeaureMemory(List<MyItem> items) throws InterruptedException {

        MemoryMeter meter = new MemoryMeter();

        int m = 16;

        double poissonLambda = 1 / Math.log(m);



        Index<Integer, float[], MyItem, Float> index =
                new HnswIndex.Builder<>(DistanceFunctions::cosineDistance, new IntegerMurmur3LevelAssigner(poissonLambda), items.size())
                        .setM(m)
                        .build();

        index.addAll(items);

        return meter.measureDeep(index);
    }


    private static List<MyItem> loadItemsToIndex() {
        return generateRandomItems(500_000, 90);
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
            result[i] = ThreadLocalRandom.current().nextFloat();
        }
        return result;
    }
}
