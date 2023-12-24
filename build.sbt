import Path.relativeTo
import sys.process.*

ThisBuild / organization := "com.github.jelmerk"
ThisBuild / scalaVersion := "2.12.18"

ThisBuild / fork := true

ThisBuild / dynverSonatypeSnapshots := true
ThisBuild / publishMavenStyle := true

ThisBuild / versionScheme := Some("early-semver")

ThisBuild / Compile / doc / javacOptions ++= {
  Seq("-Xdoclint:none")
}

val java8Home = sys.env.getOrElse("JAVA_HOME_8_X64", s"${sys.props("user.home")}/.sdkman/candidates/java/8.0.382-amzn")

lazy val publishSettings = Seq(
  pomIncludeRepository := { _ => false },
  publishTo := {
    val nexus = "https://oss.sonatype.org/"
    if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
    else Some("releases" at nexus + "service/local/staging/deploy/maven2")
  },

  licenses := Seq("Apache License 2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html")),

  homepage := Some(url("https://github.com/jelmerk/hnswlib")),

  scmInfo := Some(ScmInfo(
    url("https://github.com/jelmerk/hnswlib.git"),
    "scm:git@github.com:jelmerk/hnswlib.git"
  )),

  developers := List(
    Developer("jelmerk", "Jelmer Kuperus", "jkuperus@gmail.com", url("https://github.com/jelmerk"))
  ),

  ThisBuild / credentials += Credentials(
    "Sonatype Nexus Repository Manager",
    "oss.sonatype.org",
    sys.env.getOrElse("NEXUS_USER", ""),
    sys.env.getOrElse("NEXUS_PASSWORD", "")
  )
)

lazy val noPublishSettings =
  publish / skip := true

val scalaTestVersion = "3.2.17"
val junitVersion     = "5.5.2"
val hamcrestVersion  = "2.1"
val mockitoVersion   = "3.0.0"

val sparkVersion = settingKey[String]("Spark version")

lazy val pyTest    = taskKey[Unit]("Run the python tests")
lazy val pyPublish = taskKey[Unit]("Publish the python sources to a pypi repo")

lazy val root = (project in file("."))
  .aggregate(hnswlibUtils, hnswlibCore, hnswlibCoreJdk17, hnswlibMetricsDropwizard, hnswlibScala, hnswlibSpark)
  .settings(noPublishSettings)

lazy val hnswlibUtils = (project in file("hnswlib-utils"))
  .settings(
    name             := "hnswlib-utils",
    autoScalaLibrary := false,
    crossPaths       := false,
    publishSettings,
    Compile / compile / javacOptions ++= Seq(
      "-source", "8",
      "-target", "8"
    ),
    libraryDependencies ++= Seq(
      "org.hamcrest"      % "hamcrest-library"     % hamcrestVersion                  % Test,
      "org.junit.jupiter" % "junit-jupiter-engine" % junitVersion                     % Test,
      "org.junit.jupiter" % "junit-jupiter-api"    % junitVersion                     % Test,
      "net.aichler"       % "jupiter-interface"    % JupiterKeys.jupiterVersion.value % Test
    )
  )

lazy val hnswlibCore = (project in file("hnswlib-core"))
  .dependsOn(hnswlibUtils % "test->compile")
  .settings(
    name             := "hnswlib-core",
    autoScalaLibrary := false,
    crossPaths       := false,
    publishSettings,
    Compile / compile / javacOptions ++= Seq(
      "-source", "8",
      "-target", "8"
    ),
    libraryDependencies ++= Seq(
      "org.eclipse.collections" % "eclipse-collections"  % "9.2.0",
      "org.hamcrest"            % "hamcrest-library"     % hamcrestVersion                  % Test,
      "org.junit.jupiter"       % "junit-jupiter-engine" % junitVersion                     % Test,
      "org.junit.jupiter"       % "junit-jupiter-api"    % junitVersion                     % Test,
      "net.aichler"             % "jupiter-interface"    % JupiterKeys.jupiterVersion.value % Test
    )
  )

lazy val hnswlibCoreJdk17 = (project in file("hnswlib-core-jdk17"))
  .dependsOn(hnswlibCore)
  .settings(
    name             := "hnswlib-core-jdk17",
    autoScalaLibrary := false,
    crossPaths       := false,
    publishSettings,
    Compile / compile / javacOptions ++= Seq(
      "-source", "17",
      "-target", "17",
      "--enable-preview",
      "--add-modules", "jdk.incubator.vector"
    ),
    Compile / doc / javacOptions ++= Seq(
      "-source", "17",
      "--enable-preview",
      "--add-modules", "jdk.incubator.vector"
    ),
    Test / javaOptions ++= Seq(
      "--enable-preview",
      "--add-modules", "jdk.incubator.vector"
    ),
    libraryDependencies ++= Seq(
      "org.hamcrest"      % "hamcrest-library"     % hamcrestVersion                  % Test,
      "org.junit.jupiter" % "junit-jupiter-engine" % junitVersion                     % Test,
      "org.junit.jupiter" % "junit-jupiter-api"    % junitVersion                     % Test,
      "net.aichler"       % "jupiter-interface"    % JupiterKeys.jupiterVersion.value % Test
    )
  )

lazy val hnswlibMetricsDropwizard = (project in file("hnswlib-metrics-dropwizard"))
  .dependsOn(hnswlibCore)
  .settings(
    name             := "hnswlib-metrics-dropwizard",
    autoScalaLibrary := false,
    crossPaths       := false,
    publishSettings,
    Compile / compile / javacOptions ++= Seq(
      "-source", "8",
      "-target", "8"
    ),
    libraryDependencies ++= Seq(
      "io.dropwizard.metrics" % "metrics-core"          % "4.1.0",
      "org.awaitility"        % "awaitility"            % "4.0.1"                          % Test,
      "org.mockito"           % "mockito-junit-jupiter" % mockitoVersion                   % Test,
      "org.mockito"           % "mockito-core"          % mockitoVersion                   % Test,
      "org.hamcrest"          % "hamcrest-library"      % hamcrestVersion                  % Test,
      "org.junit.jupiter"     % "junit-jupiter-engine"  % junitVersion                     % Test,
      "org.junit.jupiter"     % "junit-jupiter-api"     % junitVersion                     % Test,
      "net.aichler"           % "jupiter-interface"     % JupiterKeys.jupiterVersion.value % Test
    )
  )

lazy val hnswlibScala = (project in file("hnswlib-scala"))
  .dependsOn(hnswlibCore)
  .dependsOn(hnswlibMetricsDropwizard % Optional)
  .settings(
    name               := "hnswlib-scala",
    crossScalaVersions := List("2.11.12", "2.12.18", "2.13.10"),
    publishSettings,
    scalacOptions := Seq(
      "-target:jvm-1.8",
      "-encoding", "UTF-8"
    ),
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % scalaTestVersion % Test
    )
  )

lazy val hnswlibSpark = (project in file("hnswlib-spark"))
  .dependsOn(hnswlibUtils)
  .dependsOn(hnswlibScala)
  .settings(
    name := s"hnswlib-spark_${sparkVersion.value.split('.').take(2).mkString("-")}",
    publishSettings,
    crossScalaVersions := {
      if (sparkVersion.value >= "3.2.0") {
        Seq("2.12.18", "2.13.10")
      } else if (sparkVersion.value >= "3.0.0") {
        Seq("2.12.18")
      } else {
        Seq("2.12.18", "2.11.12")
      }
    },
    javaHome := Some(file(java8Home)),
    Compile / unmanagedSourceDirectories += baseDirectory.value / "src" / "main" / "python",
    Test / unmanagedSourceDirectories += baseDirectory.value / "src" / "test" / "python",
    Compile / packageBin / mappings ++= {
      val base = baseDirectory.value / "src" / "main" / "python"
      val srcs = base ** "*.py"
      srcs pair relativeTo(base)
    },
    assembly / mainClass := None,
    assembly / assemblyOption ~= {
      _.withIncludeScala(false)
    },
    sparkVersion := sys.props.getOrElse("sparkVersion", "3.3.2"),
    pyTest := {
      val log = streams.value.log

      val artifactPath = (Compile / assembly).value.getAbsolutePath
      if (scalaVersion.value == "2.12.18" && sparkVersion.value >= "3.0.0" || scalaVersion.value == "2.11.12") {
        val pythonVersion = if (scalaVersion.value == "2.11.12") "python3.7" else "python3.9"
        val ret = Process(
          Seq("./run-pyspark-tests.sh", sparkVersion.value, pythonVersion),
          cwd = baseDirectory.value,
          extraEnv = "JAVA_HOME" -> java8Home, "ARTIFACT_PATH" -> artifactPath
        ).!
        require(ret == 0, "Python tests failed")
      } else {
        // pyspark packages support just one version of scala. You cannot use 2.13.x because it ships with 2.12.x jars
        log.info(s"Running pyTests for Scala ${scalaVersion.value} and Spark ${sparkVersion.value} is not supported.")
      }
    },
    test := {
      (Test / test).value
      (Test / pyTest).value
    },
    pyTest := pyTest.dependsOn(assembly).value,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-hive"         % sparkVersion.value             % Provided,
      "org.apache.spark" %% "spark-mllib"        % sparkVersion.value             % Provided,
      "com.holdenkarau"  %% "spark-testing-base" % s"${sparkVersion.value}_1.4.7" % Test,
      "org.scalatest"    %% "scalatest"          % scalaTestVersion               % Test
    )
  )