ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

lazy val root = (project in file("."))
  .settings(
    name := "Assignment - spark 2"
  )

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.3.22"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.3.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.3.2"

