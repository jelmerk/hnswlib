package com.github.jelmerk.knn.metrics;

import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.ReadOnlyIndex;
import com.github.jelmerk.knn.SearchResult;
import org.eclipse.collections.impl.list.mutable.primitive.DoubleArrayList;

import java.io.Serializable;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public class StatisticsDecorator<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

    private final Index<TId, TVector, TItem, TDistance> delegate;
    private final ReadOnlyIndex<TId, TVector, TItem, TDistance> groundTruth;
    private final int sampleFrequency;

    private AtomicLong addCount = new AtomicLong();
    private AtomicLong removeCount = new AtomicLong();
    private AtomicLong searchCount = new AtomicLong();

    private final MovingAverageAccuracyCalculator accuracyEvaluator;

    private ExecutorService executorService;

    public StatisticsDecorator(Index<TId, TVector, TItem, TDistance> delegate,
                               ReadOnlyIndex<TId, TVector, TItem, TDistance> groundTruth,
                               int accuracySampleFrequency) {

        this.delegate = delegate;
        this.groundTruth = groundTruth;
        this.sampleFrequency = accuracySampleFrequency;

        this.executorService = Executors.newSingleThreadExecutor();

        this.accuracyEvaluator = new MovingAverageAccuracyCalculator(10, 1000);
        this.executorService.submit(accuracyEvaluator);
    }

    @Override
    public void add(TItem item) {
        addCount.getAndIncrement();
        delegate.add(item);
    }

    @Override
    public boolean remove(TId id) {
        removeCount.getAndIncrement();
        return delegate.remove(id);
    }

    @Override
    public int size() {
        return delegate.size();
    }

    @Override
    public Optional<TItem> get(TId id) {
        return delegate.get(id);
    }

    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {
        List<SearchResult<TItem, TDistance>> searchResults = delegate.findNearest(vector, k);

        if (searchCount.getAndIncrement() % sampleFrequency == 0) {
            accuracyEvaluator.offer(new RequestAndResults(vector, k, searchResults));
        }

        return searchResults;
    }


    class MovingAverageAccuracyCalculator implements Runnable {

        private final int samples;
        private final ArrayBlockingQueue<RequestAndResults> queue;
        private final DoubleArrayList results;

        private volatile boolean running = true;
        private volatile double averagePrecision;

        MovingAverageAccuracyCalculator(int maxBacklog, int samples) {
            this.samples = samples;

            this.queue  = new ArrayBlockingQueue<>(maxBacklog);
            this.results = new DoubleArrayList(samples);
        }

        @Override
        public void run() {
            try {
                while (running) {
                    RequestAndResults item = queue.poll(500, TimeUnit.MILLISECONDS);

                    if (item != null) {
                        List<SearchResult<TItem, TDistance>> expectedResults = groundTruth.findNearest(item.vector, item.k);

                        int correct = expectedResults.stream().mapToInt(r -> item.searchResults.contains(r) ? 1 : 0).sum();
                        double precision = (double) correct / (double) expectedResults.size();

                        if (results.size() >= samples) {
                            results.removeAtIndex(0);
                        }

                        results.add(precision);

                        averagePrecision = results.sum() / results.size();
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
               // just die
            }
        }

        boolean offer(RequestAndResults requestAndResults) {
            return queue.offer(requestAndResults); // won't block if we can't keep up but will return false
        }

        public double getAveragePrecision() {
            return averagePrecision;
        }

        void shutdown() {
            running = false;
        }
    }

    class RequestAndResults {
        TVector vector;
        int k;
        List<SearchResult<TItem, TDistance>> searchResults;

        RequestAndResults(TVector vector, int k, List<SearchResult<TItem, TDistance>> searchResults) {
            this.vector = vector;
            this.k = k;
            this.searchResults = searchResults;
        }
    }
}
