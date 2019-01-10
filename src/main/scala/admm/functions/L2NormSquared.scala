package admm.functions

import breeze.optimize.{CachedDiffFunction, LBFGS}
import org.apache.spark.rdd.RDD
import admm.linalg.BlockMatrix
import breeze.linalg._
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import org.apache.spark.SparkContext
import org.apache.spark.storage._
import org.apache.spark.broadcast._

class L2NormSquared(val A: BDM[Double],
                    val b: BDV[Double],
                    var rho: Double) extends Function1[BDV[Double],Double] with Prox with Serializable{

  val numParam = A.cols
  val numSamples = A.rows
  var U = BDV.zeros[Double](numParam)
  var X = BDV.zeros[Double](numParam)
  var R = 0.0

  lazy val factor = {
    inv(A.t*A + BDM.eye[Double](A.cols) * rho)
  }

  lazy val auxUpdtX = {
    A.t*b
  }

  ///////////////////////////////////////////////

  def updateAll(Z: BDV[Double],iter:Int,newRho: Double): (BDV[Double],BDV[Double],Double) ={
    updtR(Z)
    updtU(Z)
    if (math.abs(rho-newRho)>0.01){
      U *= (rho/newRho)
      rho = newRho
    }else {

    }
    updtX(Z,rho,iter)
    (U,X,R)
  }

  def updtX(Z: BDV[Double] ,rho: Double,iters: Int): Unit = {


    X = (factor * (auxUpdtX + (Z - U)*rho))
  }




  def updtU(inval: BDV[Double]): Unit = {
    U = U + X - inval
  }

  def updtR(inval: BDV[Double]): Unit = {
    R = math.pow(norm(X - inval),2)
  }

  def changeU(inval: Double): Unit = {
    U *= inval
  }

  def changeRho(inval: Double): Unit = {
    rho = inval
  }
  ///////////////////////////////////////////////
  
  def prox(x: BDV[Double], rho: Double): BDV[Double] = {
    (factor * (A.t*b + x*rho)).toDenseVector
  }

  def apply(x: BDV[Double]): Double = {
    Math.pow(norm(A*x - b),2.0)
  }



}

object L2NormSquared {
  def fromTextFile(file: RDD[String], rho: Double, blockHeight: Int = 1024): RDF[L2NormSquared] = {
    val fns = new BlockMatrix(file, blockHeight).blocks.
      map(X => new L2NormSquared(X(::, 0 to -2), X(::,-1).toDenseVector, rho))
    new RDF[L2NormSquared](fns, 0L)
  }

  def fromMatrix(A: RDD[BDM[Double]], rho: Double): RDF[L2NormSquared] = {
    val localx = BDV.rand[Double](A.first.cols)
    val x = A.context.broadcast(localx)
    val fns = A.map(X => new L2NormSquared(X, X*x.value, rho))
    new RDF[L2NormSquared](fns, 0L)
  }

  def fromTextFileWithoutLabels(file: RDD[String], rho: Double, blockHeight: Int = 1000, model : BDV[Double]): RDF[L2NormSquared] = {
    val wtf = new BlockMatrix(file, blockHeight).blocks
    val fns = wtf.map(X => new L2NormSquared(X,X*model,rho))
    new RDF[L2NormSquared](fns, 0L)
  }

}
