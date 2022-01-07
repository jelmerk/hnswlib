Releasing hnswlib
=================

Releasing a new version to maven central is a bit cumbersome because maven does not play nice with cross releasing scala libraries

1. run export GPG_TTY=$(tty)

2. update the version number to the release version in every pom

3. commit
        
       git commit -am "Prepare release"
       
4. tag the release

       git tag v0.x.x HEAD
        
5. ./crossbuild.sh clean deploy -DperformRelease=true
   
6. release the pyspark module with

   rm -rf build ; rm -rf dist ; rm -rf pyspark_hnsw.egg-info ; python2.7 setup.py clean --all bdist_wheel && python2.7 -m twine upload dist/*
   rm -rf build ; rm -rf dist ; rm -rf pyspark_hnsw.egg-info ; python3.7 setup.py clean --all bdist_wheel && python3.7 -m twine upload dist/*
    
7. update the version number to the development version version in every pom to the new development version

8. update the version in hnswlib-pyspark/setup.py

9. commit

       git commit -am "Next development version"
       
10. push 

       git push
       git push --tags
       
