package org.github.jelmerk.hnsw;


import java.io.Serializable;

/**
 * Cache for distance between 2 points.
 *
 */
class DistanceCache implements Serializable {

    /**
     * https://referencesource.microsoft.com/#mscorlib/system/array.cs,2d2b551eabe74985,references
     * We use powers of 2 for efficient modulo
     * 2^28 = 268435456
     * 2^29 = 536870912
     * 2^30 = 1073741824
     */
    private static final int MAX_ARRAY_LENGTH = 0x40000000;

    /**
     * The cached values.
     */
    private float[] values;

    /**
     * The cached keys;
     */
    private long[] keys;

    /**
     * Initializes a new instance of the {@link DistanceCache} class.
     *
     * @param pointsCount The number of points to allocate cache for.
     */
    @SuppressWarnings({"unchecked"})
    DistanceCache(int pointsCount) {

        long capacity = ((long) pointsCount * (pointsCount + 1)) >> 1;
        capacity = capacity < MAX_ARRAY_LENGTH ? capacity : MAX_ARRAY_LENGTH;

        this.keys = new long[(int) capacity];
        this.values = new float[(int) capacity];


        // TODO: may be there is a better way to warm up cache and force OS to allocate pages
        for (int i = 0; i < this.keys.length; i++) {
            this.keys[i] = -1;
            this.values[i] = 0;
        }
    }

    /**
     * Tries to get value from the cache.
     *
     * @param fromId The 'from' point identifier.
     * @param toId The 'to' point identifier.
     * @return True if the distance value is retrieved from the cache.
     */
    float getValueOrDefault(int fromId, int toId, float defaultValue)  {
        long key = makeKey(fromId, toId);
        int hash = (int)(key & (MAX_ARRAY_LENGTH - 1));

        if (this.keys[hash] == key) {
            return this.values[hash];
        }

        return defaultValue;
    }

    /**
     * Caches the distance value.
     *
     * @param fromId The 'from' point identifier.
     * @param toId The 'to' point identifier.
     * @param distance The distance value to cache.
     */
    void setValue(int fromId, int toId, float distance) {
        long key = makeKey(fromId, toId);
        int hash = (int)(key & (MAX_ARRAY_LENGTH - 1));
        this.keys[hash] = key;
        this.values[hash] = distance;
    }

    /**
     * Builds key for the pair of points.
     *
     * @param fromId The from point identifier.
     * @param toId The to point identifier.
     * @return Key of the pair.
     */
    private static long makeKey(int fromId, int toId) {
        return fromId > toId
                ? (((long)fromId * (fromId + 1)) >> 1) + toId
                : (((long)toId * (toId + 1)) >> 1) + fromId;
    }

}