ThisBuild / organization := "com.github.jelmerk"
ThisBuild / scalaVersion := "2.12.18"

ThisBuild / fork := true

ThisBuild / dynverSonatypeSnapshots := true

ThisBuild / versionScheme := Some("early-semver")

ThisBuild / Compile / doc / javacOptions ++= {
  Seq("-Xdoclint:none")
}

lazy val publishSettings = Seq(
  pomIncludeRepository := { _ => false },

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
  ),

  publishTo := sonatypePublishToBundle.value,
  sonatypeSessionName := s"[sbt-sonatype] ${name.value} ${version.value}"

)

lazy val noPublishSettings =
  publish / skip := true

val scalaTestVersion = "3.2.17"
val junitVersion     = "5.5.2"
val hamcrestVersion  = "2.1"
val mockitoVersion   = "3.0.0"

lazy val root = (project in file("."))
  .aggregate(hnswlibUtils, hnswlibCore, hnswlibCoreJdk17, hnswlibMetricsDropwizard, hnswlibScala)
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
  .dependsOn(hnswlibCoreJdk17 % Optional)
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