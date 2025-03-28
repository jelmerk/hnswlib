
Frequently asked questions
--------------------------

- Will [SIMD](https://en.wikipedia.org/wiki/SIMD) instructions be used ?

  It depends on the jvm implementation because until project [JEP-338](https://openjdk.java.net/jeps/338) is completed you 
  cannot use SIMD explicitly from java. With the oracle / open jdk you can pass the following options to view the assembly 
  code generated by the JIT 

      -XX:+UseSuperWord -XX:+UnlockDiagnosticVMOptions -XX:CompileCommand=print,*FloatCosineDistance.distance

  For more information consult [Vectorization in HotSpot JVM](https://cr.openjdk.java.net/~vlivanov/talks/2017_Vectorization_in_HotSpot_JVM.pdf)


- How much memory is used?

  Rather than providing you with a complicated formula that takes many variables into account. I suggest 
  using using [Java Agent for Memory Measurements](https://github.com/jbellis/jamm) to measure actual object
  memory use including JVM overhead. Here's an example of how to do this :
  
        import org.github.jamm.MemoryMeter;
        import com.github.jelmerk.hnswlib.core.DistanceFunctions;
        import com.github.jelmerk.hnswlib.core.Index;
        
        import java.util.List;
        
        public class MemoryMeasurement {
       
            public static void main(String[] args) throws Exception {
                List<MyItem> allElements = loadItemsToIndex();
        
                int increment = 100_000;
                long lastSeenMemory = -1L;
        
                for (int i = increment; i <= allElements.size(); i += increment) {
                    List<MyItem> items = allElements.subList(0, i);
        
                    long memoryUsed = createIndexAndMeasureMemory(items);
        
                    if (lastSeenMemory == -1) {
                        System.out.printf("Memory used for index of size %d is %d bytes%n", i, memoryUsed);
                    } else {
                        System.out.printf("Memory used for index of size %d is %d bytes, delta with last generated index : %d bytes%n", i, memoryUsed, memoryUsed - lastSeenMemory);
                    }
                    
                    lastSeenMemory = memoryUsed;
                    createIndexAndMeaureMemory(items);
                }
            }
        
            private static long createIndexAndMeasureMemory(List<MyItem> items) throws InterruptedException {
                MemoryMeter meter = new MemoryMeter();

                Index<String, float[], MyItem, Float> index = HnswIndex
                    .newBuilder(DistanceFunctions.FLOAT_COSINE_DISTANCE, items.size())
                        .withM(16)
                        .build();

                index.addAll(items);
                
                return meter.measureDeep(index);
            }
         }
 
   Run the above code with -javaagent:/path/to/jamm-0.3.0.jar 
   
   The output of this program will show approximately how much memory adding an additional 100.000 elements to this index will take up
   Since the amount of memory used scales roughly linearly with the amount of elements you should be able to work out your memory requirements 
   

- How do I measure the precision of the index ?

  By calling asExactIndex on the hnswlib index you create a view on the HnswIndex that produces exact results.
  Which you can use to compare the resuls of the approximative index with
  
  
        HnswIndex<String, float[], Word, Float> hnswIndex = HnswIndex
                .newBuilder(DistanceFunctions.FLOAT_COSINE_DISTANCE, words.size())
                .build();
        hnswIndex.addAll(words);

        Index<String, float[], Word, Float> groundTruthIndex = hnswIndex.asExactIndex();

        List<SearchResult<Word, Float>> expectedResults = groundTruthIndex.findNeighbors("king", 10);
        List<SearchResult<Word, Float>> actualResults = hnswIndex.findNeighbors("king", 10);

        int correct = expectedResults.stream().mapToInt(r -> actualResults.contains(r) ? 1 : 0).sum();
        double precision = (double) correct / (double) expectedResults.size();

        System.out.printf("Precision @10 : %f%n", precision);


  If the precision is not what you expect take a look at javadoc of the parameters of the hnsw index builder.
    
