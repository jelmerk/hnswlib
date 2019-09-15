Releasing hnswlib
=================

Releasing a new version to maven central is a bit cumbersome because maven does not play nice with cross releasing scala libraries

1. run export GPG_TTY=$(tty)

2. update the version number to the release version in every pom

3. commit
        
       git commit -am "Prepare release"
       
4. tag the release

       git tag v0.x.x head
        
5. release for scala 2.11 by running the following command from the root of the project 
 
       mvn clean deploy -Dscala-2.11 -DperformRelease=true
       
6. to release hnswlib-scala for scala 2.12
 
   in hnswlib-scala/pom.xml change the artifact id from hnswlib-scala_2.11 to hnswlib-scala_2.12 in
   
   then he hnswlib-scala folder run mvn clean deploy -Dscala-2.12 -DperformRelease=true
   
   finally revert the change to the pom
       
7. to release hnswlib-spark for other versions

   in hnswlib-spark/pom.xml change the artifact id from hnswlib-spark_2.3.0_2.11 to hnswlib-spark_2.4.0_2.11
   
   then he hnswlib-spark folder run mvn clean deploy -Dscala-2.11 -Dspark-2.4 -DperformRelease=true 
   
   in hnswlib-spark/pom.xml change the artifact id from hnswlib-spark_2.4.0_2.11 to hnswlib-spark_2.4.0_2.12
   
   then he hnswlib-spark folder run mvn clean deploy -Dscala-2.12 -Dspark-2.4 -DperformRelease=true

   finally revert the change to the pom
   
8. release the pyspark module with

   python2.7 setup.py bdist_wheel && python2.7 -m twine upload dist/*
   python3.7 setup.py bdist_wheel && python3.7 -m twine upload dist/*
    
8. update the version number to the development version version in every pom to the new development version

9. update the version in hnswlib-pyspark/setup.py

10. commit

       git commit -am "Next development version"
       
11. push 

       git push
       git push --tags
       
