//package org.apache.spark.mllib.linalg.LogRegressionXLocal
package admm.functions

import java.nio.ByteBuffer

import admm.functions.{ValueAndGradientLogReg, RDF, Prox}
import breeze.optimize.{CachedDiffFunction, LBFGS => BreezeLBFGS}
import bungee.LBFGS
import org.apache.spark.rdd.RDD
import admm.linalg.BlockMatrix
import breeze.linalg._
import breeze.linalg.{Vector => BV, DenseVector => BDV, DenseMatrix => BDM}
import org.apache.spark.mllib.linalg.{DenseVector => NDV, Vector => NV, CostFuncX, Vectors}
import bungee._
import org.apache.spark.mllib.util.MLUtils


class LogRegressionXLocal(data: BDM[Double],
                          numCorrectionsLBFGS: Int,
                          lbfgs_tolerance: Double,
                          var rho: Double,
                          alpha: Double) extends Function1[BDV[Double],Double] with Prox  with Serializable{



  val numParam = data.cols
  val numSamples = data.rows
  //var timeEvalF = 0.0
  //var timeLBFGS = 0.0
  //var numIter = 0
  //var lbfgsTolerance = 1.0
  var U = BDV.zeros[Double](numParam)
  var X = BDV.zeros[Double](numParam)
  var Xhat = BDV.zeros[Double](numParam)
  var R = 0.0

  val costFun = new CostFuncX(data,BDV.zeros[Double](11),U,rho)
  val localLBFGS = new LBFGS
  //val gradienteYPerdida = new ValueAndGradientLogReg(data)


  def updateAll(Z: BDV[Double],iter:Int,newRho: Double): (BDV[Double],BDV[Double],Double) ={
    //costFun.updateU(U)
    //costFun.updateZ(Z)
    //updtX(Z,rho,iter)
    updtR(Z)
    updtU(Z)
    if (math.abs(rho-newRho)>0.0000000001){
      U *= (rho/newRho)
      rho = newRho
      costFun.updateRho(newRho)
    }
     //Version Original "Correcta" que funciona a rho constante

    costFun.updateU(U)
    costFun.updateZ(Z)
    updtX(Z,rho,iter)

    Xhat = alpha*X + (1-alpha)*Z

    (U,Xhat,R)
  }



  def prox(inval: BDV[Double] ,rho: Double): BDV[Double] = {
    /*
    val UminusZ = U-inval
    var pesos = new Array[Double](numParam)
    val diago = new Array[Double](numParam)
    diago.map(x => 1)

    val iprint = new Array[Int](2)
    iprint(0)= -1
    iprint(1)=0
    var iflag = new Array[Int](1)
    iflag(0) = 0
    var eval=(0.0,toBreeze(pesos).toDenseVector)
    var numiter=0

    do{
      numiter = numiter+1
      eval=gradienteYPerdida.evaluate(toBreeze(pesos).toDenseVector,UminusZ)
      localLBFGS.lbfgs(numParam, 7, pesos,eval._1,eval._2.toArray,false,diago,iprint,0.0001,2.220446049250313E-16,iflag)
    }while (iflag(0)==1)
    X = toBreeze(pesos).toDenseVector
    */
    X
  }


  def updtX(inval: BDV[Double] ,rho: Double,iters: Int): Unit = {

    val lbfgs   = new BreezeLBFGS[BDV[Double]](-1, numCorrectionsLBFGS, lbfgs_tolerance)
    val states  = lbfgs.iterations(new CachedDiffFunction(costFun), X)
    var step = states.next()
    while(states.hasNext) {
      step = states.next()
    }
    X = step.x
  }




  def updtU(inval: BDV[Double]): Unit = {
    U = U + Xhat - inval
    //quitar el hat despues
  }

  def updtR(inval: BDV[Double]): Unit = {
    R = math.pow(norm(X - inval),2)
  }

  def changeU(inval: Double): Unit = {
    U *= inval
    costFun.updateU(U)
  }

  def changeRho(inval: Double): Unit = {
    rho = inval
    costFun.updateRho(inval)
  }

  def apply(x: BDV[Double]): Double = {
    Math.pow(0,2.0)
  }

  def fromBreeze(breezeVector: BV[Double]): NV = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new NDV(v.data)
        } else {
          new NDV(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  def toBreeze(values: Array[Double]): BV[Double] = new BDV[Double](values)

}



object LogRegressionXLocal {

  def fromTextFileWithoutLabels(file: RDD[String], rho: Double, blockHeight: Int = 1000,numberOfCorrectionsLBFGS: Int,toleranceLBFGS: Double , alpha: Double, model : BDV[Double]): RDF[LogRegressionXLocal] = {
    val wtf = new BlockMatrix(file, blockHeight).blocks.map(x => getlabels(x,model) )
    val fns = wtf.map(X => new LogRegressionXLocal(X, numberOfCorrectionsLBFGS,toleranceLBFGS,rho,alpha))
    new RDF[LogRegressionXLocal](fns, 0L)
  }

  def fromBinFileWithLabels(file: RDD[BDM[Double]], rho: Double, blockHeight: Int = 1000,numberOfCorrectionsLBFGS: Int,toleranceLBFGS: Double ,alpha: Double, model : BDV[Double]): RDF[LogRegressionXLocal] = {
    val fns = file.map(X => new LogRegressionXLocal(X, numberOfCorrectionsLBFGS,toleranceLBFGS,rho,alpha))
    new RDF[LogRegressionXLocal](fns, 0L)
  }


  def getlabels(matrix: BDM[Double], model: BDV[Double]): BDM[Double] = {
    val n = matrix.rows
    val m = matrix.cols
    //println("Tengo estas filas--->" + n)
    (for(x: Int <- Range(0,n-1)) {
      var label = math.signum(matrix(x, ::) * model)
      for (j: Int <- Range(0, m - 2)) {//Add noise in a deterministic way
        matrix(x, j) = matrix(x, j) + 0.2 * matrix(n - 1 - x, j)
      }
      matrix(x,-1) = if(label!=0) label else 1
    })

    //println("Labels............." + matrix(::,-1).toString())
    //println("Models............." + model.toString())

    //println("numBlockMatrix->" + n)
    if(sum(matrix(::,-1))== matrix.rows) System.exit(20)
    if(sum(matrix(::,-1))== -1*matrix.rows) System.exit(40)
    if(n<10) System.exit(60+n)
    matrix
  }


}
