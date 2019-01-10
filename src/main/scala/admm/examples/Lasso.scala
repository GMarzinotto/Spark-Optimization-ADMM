/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package admm.examples


import java.nio.ByteBuffer

import com.typesafe.config.ConfigFactory
import admm.functions.LogRegressionXLocal
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *}
import admm.functions._
import admm.solver._


object Lasso {

  case class Params(
       inputFileName: String = "",           //If input file is used, Input file path and name
       lambda: Double = 0.01,                //Value of the lasso regularization
       blockheight: Int = 100,              //Size to split the file or the random generated data
       maxiters: Int = 120,                  //Maximal number of iterations
       rho: Double = 1.0,                    //Initial rho value
       rhoCorrectionFactor: Double = 2.0,    //Scale factor applied to rho when the condition is satisfied. To avoid updating rho set this value to 1.0
       rhoCorrectionMu: Double = 10.0,       //Constant associated to the rho correction condition. To avoid updating rho this can be set to a very large value
       abstol: Double = 1e-4,                //Absolute Tolerance
       reltol: Double = 1e-3,                //Relative Tolerance
       alpha: Double = 1.0,                  //Alpha used for relaxation. To avoid relaxation set alpha to 1.0
       numberOfCorrectionsLBFGS: Int = 3,    //Number of corrections used in the L-BFGS
       lowerBoundLBFGSres: Double = 0.01     //Minimal stop condition on the residual of L-BFGS
  )

  val defaultParams = Params()

  val useLasso = false
//"Problem1_LogisticReg"
  val ProblemFileList = List("Problem1_LogisticReg")
  val MConfigFileList = List("ConfigurationADMM")


  def main(args: Array[String]): Unit = {
    val Tiempos = ProblemFileList.map(problemFileName => {
      val probParams = ConfigFactory.load(problemFileName)
      val inputFileName   = probParams.getString("ProblemConfiguration.inputFileName")
      val lambda          = probParams.getDouble("ProblemConfiguration.lambda")
      val numModels       = probParams.getInt("ProblemConfiguration.numberOfModels")

      MConfigFileList.foreach(configurationFileName => {
        val confParams = ConfigFactory.load(configurationFileName)
        (1 to 3).map(rep => {
          val path = "ProblemConfiguration.model" + rep.toString + "Optimal"
          val params: Params = new Params(
          (inputFileName + "_Model" + rep + ".bin"),
          lambda,
          confParams.getInt("admmConfiguration.blockheight"),
          confParams.getInt("admmConfiguration.maxiters"),
          confParams.getDouble("admmConfiguration.rho"),
          confParams.getDouble("admmConfiguration.rhoCorrectionFactor"),
          confParams.getDouble("admmConfiguration.rhoCorrectionMu"),
          confParams.getDouble("admmConfiguration.abstol"),
          confParams.getDouble("admmConfiguration.reltol"),
          confParams.getDouble("admmConfiguration.alpha"),
          confParams.getInt("admmConfiguration.numberOfCorrectionsLBFGS"),
          confParams.getDouble("admmConfiguration.lowerBoundLBFGSres")
        )
        val model: Array[Double] = (probParams.getDoubleList(path)).toArray.map(_.toString.toDouble)
        run(params,problemFileName,configurationFileName,rep,model)})
      })
    })
  }


  def run(params: Params, problemFileName: String, configurationFileName: String, rep: Int, filemodel: Array[Double]) = {

    val inputFileName   = "/home/gabriel/Desktop/ProblemFiles/" + params.inputFileName

    val conf = new SparkConf().setMaster("local[*]").setAppName("Lasso")
    val sc = new SparkContext(conf)

    var filepath: String = ""

    if(useLasso) {
      filepath = "/home/gabriel/Results/LogisticRegression/" + problemFileName + "_Lasso" + rep + "/admmTreeA/" + configurationFileName + "/test.csv"
    } else{
      filepath = "/home/gabriel/Results/LogisticRegression/" + problemFileName + "_NoLasso" + rep + "/admmTreeA/" + configurationFileName + "/test.csv"
    }

    //var f: RDF[LogRegressionXLocal] = null
    //var f: RDF[L2NormSquared] = null

    var modelBDV = new BDV[Double](filemodel)
    var numParams: Int = 0

    val blockheight: Int = params.blockheight
    val nrecords = filemodel.length
    val bytesize = 8
    val data = sc.binaryRecords(inputFileName,nrecords*bytesize*blockheight).map(v =>   convertoDoubles(v,blockheight,nrecords)).cache()


    val t1 = System.currentTimeMillis()/1000.0
    val f = LogRegressionXLocal.fromBinFileWithLabels(data, params.rho, blockheight, params.numberOfCorrectionsLBFGS, params.lowerBoundLBFGSres,params.alpha,modelBDV)
    val t2 = System.currentTimeMillis()/1000.0
    //f = L2NormSquared.fromTextFileWithoutLabels(A,params.rho,blockheight,modelBDV)
    //Retrieve the number of parameters from the file
    numParams = f.splits.first().numParam
    f.splits.cache
    f.splits.count.toInt
    //
    //params.lambda

    val t3 = System.currentTimeMillis()/1000.0
    val g = new L1Norm(0.0,params.blockheight)
    val admm = new ConsensusADMMSolver(f, g, numParams, params.abstol, params.reltol, params.alpha,params.rho,params.rhoCorrectionFactor,
                                        params.rhoCorrectionMu, params.lowerBoundLBFGSres,modelBDV,sc) with Instrumentation
    val t4 = System.currentTimeMillis()/1000.0

    admm.solve(params.rho,params.maxiters)

    //admm.printData
    admm.printFile(filepath,((t2-t1)+(t4-t3)),modelBDV,admm.z)

    sc.stop
  }


  def convertoDoubles(arr: Array[Byte],nrows: Int, ncols: Int): BDM[Double] = {
    val arr2 = BDV[Byte](arr)
    val arr3 = (0 to arr.length/8-1).map(x => ByteBuffer.wrap((arr2((x)*8 to ((x+1)*8 -1))).toArray).getDouble).toArray
    val arr4 = BDM.create(ncols,nrows,arr3)
    arr4.t
  }


}
