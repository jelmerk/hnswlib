package com.github.jelmerk.hnswlib.core.util;


/**
 * Murmur3 is successor to Murmur2 fast non-crytographic hash algorithms.
 *
 * Murmur3 32 and 128 bit variants.
 * 32-bit Java port of https://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp#94
 * 128-bit Java port of https://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp#255
 *
 * This is a public domain code with no copyrights.
 * From homepage of MurmurHash (https://code.google.com/p/smhasher/),
 * "All MurmurHash versions are public domain software, and the author disclaims all copyright
 * to their code."
 */
public class Murmur3 {

    // Constants for 32 bit variant
    private static final int C1_32 = 0xcc9e2d51;
    private static final int C2_32 = 0x1b873593;
    private static final int R1_32 = 15;
    private static final int R2_32 = 13;
    private static final int M_32 = 5;
    private static final int N_32 = 0xe6546b64;


    private static final int DEFAULT_SEED = 104729;


    /**
     * Murmur3 32-bit variant.
     *
     * @param data - input byte array
     * @return - hashcode
     */
    public static int hash32(byte[] data) {
        return hash32(data, 0, data.length, DEFAULT_SEED);
    }

    /**
     * Murmur3 32-bit variant.
     *
     * @param data   - input byte array
     * @param offset - offset of data
     * @param length - length of array
     * @param seed   - seed. (default 0)
     * @return - hashcode
     */
    public static int hash32(byte[] data, int offset, int length, int seed) {
        int hash = seed;
        final int nblocks = length >> 2;

        // body
        for (int i = 0; i < nblocks; i++) {
            int i_4 = i << 2;
            int k = (data[offset + i_4] & 0xff)
                    | ((data[offset + i_4 + 1] & 0xff) << 8)
                    | ((data[offset + i_4 + 2] & 0xff) << 16)
                    | ((data[offset + i_4 + 3] & 0xff) << 24);

            hash = mix32(k, hash);
        }

        // tail
        int idx = nblocks << 2;
        int k1 = 0;
        switch (length - idx) {
            case 3:
                k1 ^= data[offset + idx + 2] << 16;
            case 2:
                k1 ^= data[offset + idx + 1] << 8;
            case 1:
                k1 ^= data[offset + idx];

                // mix functions
                k1 *= C1_32;
                k1 = Integer.rotateLeft(k1, R1_32);
                k1 *= C2_32;
                hash ^= k1;
        }

        return fmix32(length, hash);
    }

    private static int mix32(int k, int hash) {
        k *= C1_32;
        k = Integer.rotateLeft(k, R1_32);
        k *= C2_32;
        hash ^= k;
        return Integer.rotateLeft(hash, R2_32) * M_32 + N_32;
    }

    private static int fmix32(int length, int hash) {
        hash ^= length;
        hash ^= (hash >>> 16);
        hash *= 0x85ebca6b;
        hash ^= (hash >>> 13);
        hash *= 0xc2b2ae35;
        hash ^= (hash >>> 16);

        return hash;
    }

}