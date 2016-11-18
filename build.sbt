name := "Spark Scala Machine Learning Examples"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "2.0.1" % "provided",
	"org.apache.spark" %% "spark-sql" % "2.0.1" % "provided",
	"org.apache.spark" %% "spark-streaming" % "2.0.1" % "provided",
	"org.apache.spark" %% "spark-mllib" % "2.0.1" % "provided",
	"com.github.nscala-time" %% "nscala-time" % "1.8.0",
	"com.github.scopt" %% "scopt" % "3.5.0"
)