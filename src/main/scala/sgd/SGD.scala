/**
 * Created by gabriel on 8/11/15.
 */

import java.io.{File, PrintWriter}
import java.nio.ByteBuffer

import com.typesafe.config.ConfigFactory
import org.apache.spark.mllib.linalg.{L1UpdaterSGD, Vectors, DenseVector, Vector}
import org.apache.spark.{SparkContext, SparkConf}
import breeze.linalg.{Vector => BV, DenseVector => BDV, _}
import org.apache.spark.mllib.optimization.{GradientDescent, LogisticGradient}



object GradientDescentSGD {

  case class Params(
       inputFileName: String = "",           //If input file is used, Input file path and name
       lambda:        Double = 0.01,                //Value of the lasso regularization
       maxiters:      Int = 300,                  //Maximal number of iterations
       learningRate:  Double = 1.0                    //Initial rho value
  )


  val defaultParams = Params()

  val useLasso = false

  val ProblemFileList = List("Problem1_LogisticReg")
  val MConfigFileList = List("ConfigurationSGD")


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
            confParams.getInt("sgdConfiguration.maxiters"),
            confParams.getDouble("sgdConfiguration.learningRate")
          )
          val model: Array[Double] = (probParams.getDoubleList(path)).toArray.map(_.toString.toDouble)
          run(params,problemFileName,configurationFileName,rep,model)})
      })
    })
  }


  def run(params: Params, problemFileName: String, configurationFileName: String, rep: Int, filemodel: Array[Double]) = {


    val inputFileName   = "/home/gabriel/Desktop/ProblemFiles/" + params.inputFileName
    //val inputFileName   = "file:///home/gmarzinotto/Problems/"  + params.inputFileName
    val lambda          = params.lambda
    val numIters        = params.maxiters
    val initLearnRate   = params.learningRate
    val inputModel      = BDV(filemodel)
    //val lambda = 0.1/inputModel.length;

    val conf = new SparkConf().setMaster("local[*]").setAppName("GradientDescentSGD")
    val sc = new SparkContext(conf)
    /*
    if(sc.master.equals("local[*]")){
      sc.setCheckpointDir("/home/gabriel/foo")
    }else{
      sc.setCheckpointDir(s"hdfs://${sc.master.substring(8,sc.master.length-5)}:9000/root/scratch")
    }*/

    var filepath: String = ""

    if(useLasso) {
      filepath = "/home/gabriel/Results/LogisticRegression/" + problemFileName + "_Lasso" + rep + "/gradientMllib/" + configurationFileName + "/test.csv"
    } else{
      filepath = "/home/gabriel/Results/LogisticRegression/" + problemFileName + "_NoLasso" + rep + "/gradientMllib/" + configurationFileName + "/test.csv"
    }

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

    val training = data.map(x => (parserLabels(x))).cache()

    val initialWeightsWithIntercept = Vectors.zeros(inputModel.length)
    //println("----------------------------------------------------->" + initialWeightsWithIntercept.toString)

    val epochTime = System.nanoTime()/1000.0
    val updater = new L1UpdaterSGD()
    val gradient = new LogisticGradient()



    val (weightsWithIntercept, loss) = GradientDescent.runMiniBatchSGD(
      training,gradient,updater,0.3*math.sqrt(inputModel.length),600,lambda,1.0,initialWeightsWithIntercept)

    val finalModel = weightsWithIntercept.toString

    sc.stop()
    printFile(filepath,updater.store, updater.startTime - epochTime ,inputModel,finalModel)


    sc.stop
  }


  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }


  def getlabel(datapoint: BDV[Double], modelo: BDV[Double]): Double = {
    val label = math.signum(datapoint dot modelo)
    if(label<0) 0.0 else 1.0
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

  def convertoDoubles(arr: Array[Byte]): BDV[Double] ={
    val arr2 = BDV[Byte](arr)
    val arr3 = (0 to arr.length/8-1).map(x => ByteBuffer.wrap((arr2((x)*8 to ((x+1)*8 -1))).toArray).getDouble).toArray
    val arr4 = BDV[Double](arr3)
    arr4
  }

  def parserLabels(datapoint: BDV[Double]): (Double,Vector) = {
    val label = if(datapoint(-1) == -1) 0 else 1
    datapoint(-1) = 1
    val features = Vectors.dense(datapoint.toArray)
    (label,features)
  }

}
