name := "Spark Scala Machine Learning Examples"

version := "1.0"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "2.4.4" % "provided",
	"org.apache.spark" %% "spark-sql" % "2.4.4" % "provided",
	"org.apache.spark" %% "spark-streaming" % "2.4.4" % "provided",
	"org.apache.spark" %% "spark-mllib" % "2.4.4" % "provided",
	"com.github.nscala-time" %% "nscala-time" % "1.8.0",
	"com.github.scopt" %% "scopt" % "3.5.0"
)