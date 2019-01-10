import java.io.{File, PrintWriter}
import java.nio.ByteBuffer

import breeze.linalg.{DenseVector => BDV, Vector => BV}
import com.typesafe.config.ConfigFactory
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.classification.{LogisticRegression}  //LogisticRegression2
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

object TestOWLQN {

  case class Params(
                     inputFileName: String = "",           //If input file is used, Input file path and name
                     lambda: Double = 0.01,                //Value of the lasso regularization
                     numCorrections: Int = 10,
                     maxiters: Int = 200,                  //Maximal number of iterations
                     tolerance: Double = 1e-28                //Absolute Tolerance
                     )


  val defaultParams = Params()

  val useLasso = true

  val ProblemFileList = List("Problem1_LogisticReg","Problem2_LogisticReg","Problem3_LogisticReg","Problem4_LogisticReg")
  val MConfigFileList = List("ConfigurationOWLQN")


  def main(args: Array[String]): Unit = {
    val Tiempos = ProblemFileList.map(problemFileName => {
      val probParams = ConfigFactory.load(problemFileName)
      val inputFileName   = probParams.getString("ProblemConfiguration.inputFileName")
      val lambda          = probParams.getDouble("ProblemConfiguration.lambda")
      val numModels       = probParams.getInt("ProblemConfiguration.numberOfModels")

      MConfigFileList.foreach(configurationFileName => {
        val confParams = ConfigFactory.load(configurationFileName)
        (1 to numModels).map(rep => {
                                      val path = "ProblemConfiguration.model" + rep.toString + "Optimal"
                                      val params: Params = new Params(
                                        (inputFileName + "_Model" + rep + ".bin"),
                                        lambda,
                                        confParams.getInt("lbfgsConfiguration.numCorrections"),
                                        confParams.getInt("lbfgsConfiguration.maxiters"),
                                        confParams.getDouble("lbfgsConfiguration.tolerance")
                                      )
                                      val model: Array[Double] = (probParams.getDoubleList(path)).toArray.map(_.toString.toDouble)
                                      run(params,problemFileName,configurationFileName,rep,model)})
        })
    })
  }


  def run(params: Params, problemFileName: String, configurationFileName: String, rep: Int, filemodel: Array[Double]) = {

    val inputFileName   = "/home/gabriel/Desktop/ProblemFiles/" + params.inputFileName
    //val inputFileName   = "file:///home/gmarzinotto/Problems/" + params.inputFileName

    //val lambda          = params.lambda
    //val maxNumIters        = params.maxiters
    val maxNumIters = 200
    val numCorrections  = params.numCorrections
    val methodTolerance  = params.tolerance
    val inputModel      = BDV(filemodel)
    val lambda          = 0.1/inputModel.length

    var filepath: String = ""

    if(useLasso) {
      filepath = "/home/gabriel/Results/LogisticRegression/" + problemFileName + "_Lasso" + rep + "/OWLQN/" + configurationFileName + "/test.csv"
    } else{
      filepath = "/home/gabriel/Results/LogisticRegression/" + problemFileName + "_NoLasso" + rep + "/OWLQN/" + configurationFileName + "/test.csv"
    }

    val conf = new SparkConf().setMaster("local[*]").setAppName("Lasso")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    /*
    val data = sc.textFile(inputFileName)
    val fileData = data.map { line =>
      val parts   = line.split(',')
      (BDV(parts.map(x=>x.toDouble)))
    }
    */

    val nrecords = filemodel.length;
    val bytesize = 8;
    val data = sc.binaryRecords(inputFileName,nrecords*bytesize).map(v =>  convertoDoubles(v))


    // Prepare training data from a list of (label, features) tuples.
    val trainRDD = data.map(x => parserLabel(x)).cache()
    val training = sqlContext.createDataFrame(trainRDD).toDF("label", "features")

    val epochTime = System.nanoTime()/1000.0
    // Create a LogisticRegression instance.  This instance is an Estimator.
    val lr = new LogisticRegression()
    // Print out the parameters, documentation, and any default values.
    //println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    // We may set parameters using setter methods.
    lr.setMaxIter(maxNumIters)
      .setRegParam(lambda)
      .setTol(methodTolerance)
      .setElasticNetParam(1.0)


    // Learn a LogisticRegression model.  This uses the parameters stored in lr.
    val model1 = lr.fit(training)
    // Since model1 is a Model (i.e., a Transformer produced by an Estimator),
    // we can view the parameters it used during fit().
    // This prints the parameter (name: value) pairs, where names are unique IDs for this
    // LogisticRegression instance.
    println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)
    sc.stop()

   
  }


  def printFile(path: String,lista: List[Tuple3[Int,Double,Any]], initTime: Double, TrueModel: BDV[Double], finalModel: String): Unit ={
    val iterlist = lista.iterator
    var lista2: List[Tuple3[Int,Double,Any]] = Nil
    var timevar: List[Double] = Nil
    var timevar2: List[Double] = Nil

    while(iterlist.hasNext){
      lista2 :::= List(iterlist.next())
    }
    for(k <- 0 to lista2.length-1){
      if(k==0) timevar :::= List(lista2(0)._2)
      else timevar :::= List(lista2(k)._2 - lista2(k-1)._2)
    }
    val iterTimeList = timevar.iterator
    while(iterTimeList.hasNext){
      timevar2 :::= List(iterTimeList.next())
    }

    val writer = new PrintWriter(new File(path))
    writer.write("Initialization Time: " + initTime + "\n")
    writer.write("Expected Result (If NoLasso): " + TrueModel.toString() + "\n")
    writer.write("Final Result:   " + finalModel + "\n")
    writer.write("iteration,iterTime,Model\n")
    lista2.zip(timevar2).foreach(a => printIterDataToFile(a,writer))
    writer.close()

  }



  def printIterDataToFile(dato: Tuple2[Tuple3[Int,Double,Any],Double],writer: PrintWriter): Unit ={
    val str = dato._1._1.toString  + "," + dato._2.toString + "," + dato._1._3.toString.stripPrefix("DenseVector(").stripSuffix(")").trim + "\n"
    writer.write(str)
  }

  def parserLabel(datapoint: BDV[Double]): (Double,Vector) = {
    val label = if(datapoint(-1) == -1) 0 else 1
    val features = Vectors.dense(datapoint(0 to -2).toArray)
    (label,features)
  }

  def convertoDoubles(arr: Array[Byte]): BDV[Double] ={
    val arr2 = BDV[Byte](arr)
    val arr3 = (0 to arr.length/8-1).map(x => ByteBuffer.wrap((arr2((x)*8 to ((x+1)*8 -1))).toArray).getDouble).toArray
    val arr4 = BDV[Double](arr3)
    arr4
  }
}

