package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.*;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;

import static org.hamcrest.CoreMatchers.hasItems;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class HnswIndexTest {

    private HnswIndex<String, float[], TestItem, Float> index;

    private int maxItemCount = 100;
    private int m = 12;
    private int efConstruction = 250;
    private int ef = 20;
    private DistanceFunction<float[], Float> distanceFunction = DistanceFunctions.FLOAT_COSINE_DISTANCE;

    private TestItem item1 = new TestItem("1", new float[] { 0.0110f, 0.2341f }, 10);
    private TestItem item2 = new TestItem("2", new float[] { 0.2300f, 0.3891f }, 10);
    private TestItem item3 = new TestItem("3", new float[] { 0.4300f, 0.9891f }, 10);

    @Before
    public void setUp() {
        index = HnswIndex
                .newBuilder(distanceFunction, maxItemCount)
                .withM(m)
                .withEfConstruction(efConstruction)
                .withEf(ef)
                .withRemoveEnabled()
                .build();
    }

    @Test
    public void returnM() {
        assertThat(index.getM(), is(m));
    }

    @Test
    public void returnEf() {
        assertThat(index.getEf(), is(ef));
    }

    @Test
    public void returnEfConstruction() {
        assertThat(index.getEfConstruction(), is(efConstruction));
    }

    @Test
    public void returnMaxItemCount() {
        assertThat(index.getMaxItemCount(), is(maxItemCount));
    }

    @Test
    public void returnDistanceFunction() {
        assertThat(index.getDistanceFunction(), is(distanceFunction));
    }

    @Test
    public void returnsSize() {
        assertThat(index.size(), is(0));
        index.add(item1);
        assertThat(index.size(), is(1));
    }

    @Test
    public void addAndGet() {
        assertThat(index.get(item1.id()), is(Optional.empty()));
        index.add(item1);
        assertThat(index.get(item1.id()), is(Optional.of(item1)));
    }

    @Test
    public void returnsItems() {
        assertThat(index.items().isEmpty(), is(true));
        index.add(item1);
        assertThat(index.items().size(), is(1));
        assertThat(index.items(), hasItems(item1));
    }

    @Test
    public void removeItem() {
        index.add(item1);

        assertThat(index.remove(item1.id(), item1.version()), is(true));

        assertThat(index.size(), is(0));
        assertThat(index.items().size(), is(0));
        assertThat(index.get(item1.id()), is(Optional.empty()));

        assertThat(index.asExactIndex().size(), is(0));
        assertThat(index.asExactIndex().items().size(), is(0));
        assertThat(index.asExactIndex().get(item1.id()), is(Optional.empty()));
    }

    @Test
    public void addNewerItem() {
        TestItem newerItem = new TestItem(item1.id(), new float[0], item1.version() + 1);

        index.add(item1);
        index.add(newerItem);

        assertThat(index.size(), is(1));
        assertThat(index.get(item1.id()), is(Optional.of(newerItem)));
    }

    @Test
    public void addOlderItem() {
        TestItem olderItem = new TestItem(item1.id(), new float[0], item1.version() - 1);

        index.add(item1);
        index.add(olderItem);

        assertThat(index.size(), is(1));
        assertThat(index.get(item1.id()), is(Optional.of(item1)));
    }

    @Test
    public void removeUnknownItem() {
        assertThat(index.remove("foo", 0), is(false));
    }

    @Test
    public void removeWithOldVersionIgnored() {
        index.add(item1);

        assertThat(index.remove(item1.id(), item1.version() - 1), is(false));
        assertThat(index.size(), is(1));
    }

    @Test
    public void findNearest() throws InterruptedException {
        index.addAll(Arrays.asList(item1, item2, item3));

        List<SearchResult<TestItem, Float>> nearest = index.findNearest(item1.vector(), 10);

        assertThat(nearest, is(Arrays.asList(
                new SearchResult<>(item1, 0f, Comparator.naturalOrder()),
                new SearchResult<>(item3, 0.06521261f, Comparator.naturalOrder()),
                new SearchResult<>(item2, 0.11621308f, Comparator.naturalOrder())
        )));
    }

    @Test
    public void findNeighbors() throws InterruptedException {
        index.addAll(Arrays.asList(item1, item2, item3));

        List<SearchResult<TestItem, Float>> nearest = index.findNeighbors(item1.id(), 10);

        assertThat(nearest, is(Arrays.asList(
                new SearchResult<>(item3, 0.06521261f, Comparator.naturalOrder()),
                new SearchResult<>(item2, 0.11621308f, Comparator.naturalOrder())
        )));
    }

    @Test
    public void addAllCallsProgressListener() throws InterruptedException {
        List<ProgressUpdate> updates = new ArrayList<>();

        index.addAll(Arrays.asList(item1, item2, item3), 1,
                (workDone, max) -> updates.add(new ProgressUpdate(workDone, max)), 2);

        assertThat(updates, is(Arrays.asList(
                new ProgressUpdate(2, 3),
                new ProgressUpdate(3, 3)  // emitted because its the last element
        )));
    }

    @Test
    public void saveAndLoadIndex() throws IOException {
        ByteArrayOutputStream in = new ByteArrayOutputStream();

        index.add(item1);
        index.save(in);

        in.flush();

        HnswIndex<String, float[], TestItem, Float> loadedIndex =
                HnswIndex.load(new ByteArrayInputStream(in.toByteArray()));

        assertThat(loadedIndex.size(), is(1));
    }
}
