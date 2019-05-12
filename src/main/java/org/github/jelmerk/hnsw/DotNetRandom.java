package org.github.jelmerk.hnsw;

import java.io.Serializable;

class DotNetRandom implements Serializable {
    private static final int MBIG = Integer.MAX_VALUE;
    private static final int MSEED = 161803398;

    private int inext, inextp;
    private int[] seedArray = new int[56];

    DotNetRandom(int seed) {
        int ii;
        int mj, mk;

        mj = MSEED - Math.abs(seed);
        seedArray[55] = mj;
        mk = 1;
        for (int i = 1; i < 55; i++) {
            /*
             * Apparently the range [1..55] is special (Knuth) and so we're
             * wasting the 0'th position.
             */
            ii = (21 * i) % 55;
            seedArray[ii] = mk;
            mk = mj - mk;
            if (mk < 0) {
                mk += MBIG;
            }
            mj = seedArray[ii];
        }
        for (int k = 1; k < 5; k++) {
            for (int i = 1; i < 56; i++) {
                seedArray[i] -= seedArray[1 + (i + 30) % 55];
                if (seedArray[i] < 0) {
                    seedArray[i] += MBIG;
                }
            }
        }
        inext = 0;
        inextp = 21;
    }


    synchronized double nextDouble() {
        int retVal;
        int locINext = inext;
        int locINextp = inextp;
        if (++locINext >= 56) {
            locINext = 1;
        }
        if (++locINextp >= 56) {
            locINextp = 1;
        }
        retVal = seedArray[locINext] - seedArray[locINextp];
        if (retVal < 0) {
            retVal += MBIG;
        }
        seedArray[locINext] = retVal;
        inext = locINext;
        inextp = locINextp;
        /*
         * Including this division at the end gives us significantly improved
         * random number distribution.
         */
        return (retVal * (1.0 / MBIG));
    }

}
