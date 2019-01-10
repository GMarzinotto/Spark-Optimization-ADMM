name := "ADMMTry"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.0"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.4.0"

libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.4.0"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.2.0"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "com.novocode" % "junit-interface" % "0.10" % "test"

libraryDependencies += "org.scalacheck" % "scalacheck_2.10" % "1.10.0"


libraryDependencies += "com.typesafe" % "config" % "1.3.0"

libraryDependencies += "com.github.scala-incubator.io" %% "scala-io-file" % "0.4.2"



resolvers += Resolver.mavenLocal

resolvers += Resolver.sonatypeRepo("public")

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"

resolvers ++= Seq(
  "Sonatype OSS Releases"  at "http://oss.sonatype.org/content/repositories/releases/",
  "Sonatype OSS Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots/"
)



javaOptions in (Test,run) += "-Xmx12G"
