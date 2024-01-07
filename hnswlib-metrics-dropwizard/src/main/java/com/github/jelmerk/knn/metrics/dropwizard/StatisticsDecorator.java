package com.github.jelmerk.knn.metrics.dropwizard;

import com.codahale.metrics.*;
import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.SearchResult;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static com.codahale.metrics.MetricRegistry.name;

/**
 * Decorator on top of an index that will collect statistics about the index. Such as the precision of the results
 * returned by the approximative index compared to a brute force baseline.
 *
 * @param <TId> Type of the external identifier of an item
 * @param <TVector> Type of the vector to perform distance calculation on
 * @param <TItem> Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 */
public class StatisticsDecorator<TId, TVector, TItem extends Item<TId, TVector>, TDistance,
        TApproximativeIndex extends Index<TId, TVector, TItem, TDistance>,
        TGroundTruthIndex extends Index<TId, TVector, TItem, TDistance>>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

    private static final long serialVersionUID = 1L;

    private final Timer addTimer;
    private final Timer removeTimer;
    private final Timer getTimer;
    private final Timer containsTimer;
    private final Timer findNearestTimer;
    private final Timer saveTimer;

    private final Histogram accuracyHistogram;

    private final TApproximativeIndex approximativeIndex;
    private final TGroundTruthIndex groundTruthIndex;

    private final int sampleFrequency;

    private final AtomicLong searchCount = new AtomicLong();

    private final AccuracyTestThread accuracyEvaluator;

    /**
     * Constructs a new {@link com.github.jelmerk.knn.metrics.dropwizard.StatisticsDecorator}.
     *
     * @param metricRegistry metric registry to publish the metric in
     * @param clazz the first element of the name
     * @param indexName name of the index. Will be used as part of the metric path
     * @param approximativeIndex the approximative index
     * @param groundTruthIndex the brute force index
     * @param maxAccuracySampleFrequency at most every maxAccuracySampleFrequency requests compare the results of the
     *                                   approximate index with those of the ground truth index to establish the runtime
     *                                   accuracy of the index
     */
    public StatisticsDecorator(MetricRegistry metricRegistry,
                               Class<?> clazz,
                               String indexName,
                               TApproximativeIndex approximativeIndex,
                               TGroundTruthIndex groundTruthIndex,
                               int maxAccuracySampleFrequency) {

        this.approximativeIndex = approximativeIndex;
        this.groundTruthIndex = groundTruthIndex;
        this.sampleFrequency = maxAccuracySampleFrequency;

        this.accuracyEvaluator = new AccuracyTestThread(1);

        Thread thread = new Thread(accuracyEvaluator);
        thread.setName("accuracyEvaluator");
        thread.setDaemon(true);
        thread.start();

        this.addTimer = metricRegistry.timer(name(clazz, indexName, "add"));
        this.removeTimer = metricRegistry.timer(name(clazz, indexName, "remove"));
        this.getTimer = metricRegistry.timer(name(clazz, indexName, "get"));
        this.containsTimer = metricRegistry.timer(name(clazz, indexName, "contains"));
        this.findNearestTimer = metricRegistry.timer(name(clazz, indexName, "findNearest"));
        this.saveTimer = metricRegistry.timer(name(clazz, indexName,"save"));
        this.accuracyHistogram = metricRegistry.histogram(name(clazz, indexName, "accuracy"),
                () -> new Histogram(new UniformReservoir()));

        metricRegistry.register(name(clazz, indexName, "size"), (Gauge<Integer>) approximativeIndex::size);

    }

    /**
     * Constructs a new {@link com.github.jelmerk.knn.metrics.dropwizard.StatisticsDecorator}
     *
     * @param indexName name of the index. Will be used as part of the metric path
     * @param approximativeIndex the approximative index
     * @param groundTruthIndex the brute force index
     * @param maxPrecisionSampleFrequency at most maxPrecisionSampleFrequency the results from the approximative index
     *                                    will be compared with those of the groundTruthIndex to establish the runtime
     *                                    precision of the index.
     */
    public StatisticsDecorator(String indexName,
                               TApproximativeIndex approximativeIndex,
                               TGroundTruthIndex groundTruthIndex,
                               int maxPrecisionSampleFrequency) {
        this(SharedMetricRegistries.getDefault(), StatisticsDecorator.class, indexName, approximativeIndex,
                groundTruthIndex, maxPrecisionSampleFrequency);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean add(TItem item) {
        try (Timer.Context ignored = addTimer.time()) {
            return approximativeIndex.add(item);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId id, long version) {
        try (Timer.Context ignored = removeTimer.time()) {
            return approximativeIndex.remove(id, version);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return approximativeIndex.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<TItem> get(TId id) {
        try (Timer.Context ignored = getTimer.time()) {
            return approximativeIndex.get(id);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean contains(TId id) {
        try (Timer.Context ignored = containsTimer.time()) {
            return approximativeIndex.contains(id);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<TItem> items() {
        return approximativeIndex.items();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {
        List<SearchResult<TItem, TDistance>> searchResults;

        try (Timer.Context ignored = findNearestTimer.time()) {
            searchResults = approximativeIndex.findNearest(vector, k);
        }

        if (searchCount.getAndIncrement() % sampleFrequency == 0) {
            accuracyEvaluator.offer(new RequestArgumentsAndResults(vector, k, searchResults));
        }
        return searchResults;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void save(OutputStream out) throws IOException {
        try (Timer.Context ignored = saveTimer.time()) {
            approximativeIndex.save(out);
        }
    }

    /**
     * Returns the approximative index.
     *
     * @return the approximative index
     */
    public TApproximativeIndex getApproximativeIndex() {
        return approximativeIndex;
    }

    /**
     * Returns the groundtruth index.
     *
     * @return the groundtruth index
     */
    public TGroundTruthIndex getGroundTruthIndex() {
        return groundTruthIndex;
    }

    private class AccuracyTestThread implements Runnable {

        private final ArrayBlockingQueue<RequestArgumentsAndResults> queue;

        private volatile boolean running = true;

        AccuracyTestThread(int maxBacklog) {
            this.queue  = new ArrayBlockingQueue<>(maxBacklog);
        }

        @Override
        public void run() {
            try {
                while (running) {
                    RequestArgumentsAndResults item = queue.poll(500, TimeUnit.MILLISECONDS);
                    if (item != null) {
                        List<SearchResult<TItem, TDistance>> expectedResults = groundTruthIndex.findNearest(item.vector, item.k);

                        int correct = expectedResults.stream().mapToInt(r -> item.searchResults.contains(r) ? 1 : 0).sum();
                        double precision = (double) correct / (double) expectedResults.size();
                        accuracyHistogram.update(Math.round(precision * 100));
                    }
                }
            } catch (InterruptedException e) {
                running = false;
                Thread.currentThread().interrupt();
            }
        }

        boolean offer(RequestArgumentsAndResults requestAndResults) {
            return queue.offer(requestAndResults); // won't block if we can't keep up but will return false
        }

    }

    private class RequestArgumentsAndResults {
        final TVector vector;
        final int k;
        final List<SearchResult<TItem, TDistance>> searchResults;

        RequestArgumentsAndResults(TVector vector, int k, List<SearchResult<TItem, TDistance>> searchResults) {
            this.vector = vector;
            this.k = k;
            this.searchResults = searchResults;
        }
    }
}
