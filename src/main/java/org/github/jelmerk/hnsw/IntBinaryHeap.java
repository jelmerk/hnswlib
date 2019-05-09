package org.github.jelmerk.hnsw;

import org.eclipse.collections.api.list.primitive.MutableIntList;

import java.util.Comparator;

class IntBinaryHeap {

    private final MutableIntList buffer;
    private final Comparator<Integer> comparator;

    /**
     * Initializes a new instance of the {@link IntBinaryHeap} class.
     *
     * @param buffer The buffer to store heap items.
     * @param comparer The comparer which defines order of items.
     */
    IntBinaryHeap(MutableIntList buffer, Comparator<Integer> comparer)  {
        if (buffer == null) {
            throw new IllegalArgumentException("buffer cannot be null");
        }

        this.buffer = buffer;
        this.comparator = comparer;
        for (int i = 1; i < this.buffer.size(); i++) {
            this.siftUp(i);
        }
    }

    /**
     * Gets the buffer of the heap.
     */
    public MutableIntList getBuffer() {
        return buffer;
    }


    /**
     * Pushes item to the heap.
     *
     * @param item The item to push.
     */
    void push(int item) {
        this.buffer.add(item);
        this.siftUp(this.buffer.size() - 1);
    }

    /**
     * Pops the item from the heap.
     *
     * @return The popped item.
     */
    int pop() {
        if (this.buffer.size() > 0) {
            int result = this.buffer.get(0);

            this.buffer.set(0, this.buffer.get(buffer.size() - 1));
            this.buffer.removeAtIndex(this.buffer.size() - 1);
            this.siftDown(0);

            return result;
        }

        throw new IllegalStateException("Heap is empty");
    }

    /**
     * Restores the heap property starting from i'th position down to the bottom given that the downstream items
     * fulfill the rule.
     *
     * @param i The position of item where heap property is violated.
     */
    private void siftDown(int i) {
        while (i < this.buffer.size()) {
            int l = (i << 1) + 1;
            int r = l + 1;
            if (l >= this.buffer.size()) {
                break;
            }

            int m = r < this.buffer.size() && this.comparator.compare(this.buffer.get(l), this.buffer.get(r)) < 0 ? r : l;
            if (this.comparator.compare(this.buffer.get(m), this.buffer.get(i)) <= 0) {
                break;
            }

            this.swap(i, m);
            i = m;
        }
    }

    /**
     * Restores the heap property starting from i'th position up to the head given that the upstream items fulfill
     * the rule.
     *
     * @param i The position of item where heap property is violated.
     */
    private void siftUp(int i) {
        while (i > 0) {
            int p = (i - 1) >> 1;
            if (this.comparator.compare(this.buffer.get(i), this.buffer.get(p)) <= 0) {
                break;
            }

            this.swap(i, p);
            i = p;
        }
    }

    /**
     * Swaps items with the specified indices.
     * @param i The first index.
     * @param j The second index.
     */
    private void swap(int i, int j) {
        int temp = this.buffer.get(i);
        this.buffer.set(i, buffer.get(j));
        this.buffer.set(j, temp);
    }
}
