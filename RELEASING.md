Releasing hnswlib
=================

Releasing a new version to maven central is a bit cumbersome because maven does not play nice with cross releasing scala libraries

0. export GPG_TTY=$(tty)

1. manually update the version number to the release version in every pom

2. commit
        
       git commit -am "Prepare release"
       
3. tag the release

       git tag v0.x.x head

3. in hnswlib-scala/pom.xml change 

       <artifactId>hnswlib-scala_${scala.binary.version}</artifactId>
        
    to
    
       <artifactId>hnswlib-scala_2.12</artifactId>
        
4. release for scala 2.12 by running the following command from the root of the project 
 
       mvn clean deploy -Dscala-2.12 -DperformRelease=true
       
5. in hnswlib-scala/pom.xml change 

       <artifactId>hnswlib-scala_2.12</artifactId>

   to
   
       <artifactId>hnswlib-scala_2.11</artifactId>
       
5. release for scala 2.11 by running the following command from the hnswlib-scala folder of the project

       mvn clean deploy -Dscala-2.11 -DperformRelease=true
    
6. manually update the version number to the new snapshot version in every pom and change the hnswlib-scala artifact id back to

       <artifactId>hnswlib-scala_${scala.binary.version}</artifactId> 
       
7. commit

       git commit -am "Next development version"
       
8. push 

       git push --tags
