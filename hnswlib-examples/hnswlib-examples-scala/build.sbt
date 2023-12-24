organization := "com.github.jelmerk"
name         := "hnswlib-examples-scala"
version      := "0.1"

scalaVersion := "2.12.18"

Compile / mainClass := Some("com.github.jelmerk.knn.examples.FastText")

libraryDependencies += "com.github.jelmerk" %% "hnswlib-scala" % "1.1.0"
