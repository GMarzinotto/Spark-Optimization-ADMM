package admm.functions

import breeze.linalg.{Vector => BV, DenseVector => BDV, DenseMatrix => BDM}
//import org.apache.spark.mllib.linalg.BLAS.{dot,axpy,scal}
import org.apache.spark.rdd.RDD
import admm.linalg.BlockMatrix
import breeze.linalg._


/**
 * Created by gabriel on 07/07/15.
 */
class ValueAndGradientLogReg (A: BDM[Double]) extends Serializable{

  val A2 = (A(*,::)).map(v => if(v(-1)==1) -v else { v(-1)=1; v })
  val blocksize = A.rows.toDouble

  def evaluate(weights: BDV[Double], UminusZ: BDV[Double],rho: Double): (Double,BDV[Double]) = {

    //println("Lables A" + A2(::,-1))

    var vectorsup = weights + UminusZ
    val normSqrdVector = sum(vectorsup.mapValues(v => v * v))

    val AweightsExp = (A2 * weights).mapValues(x => math.exp(x))
    val costf = sum((AweightsExp).mapValues(x => math.log1p(x))) + (rho / 2) * normSqrdVector
    vectorsup :*= rho
    val gradf = vectorsup + (A2.t * (AweightsExp).mapValues(x => x / (1 + x)))
    (costf, gradf)
  }

  def getInvHessianDiag(weights: BDV[Double], rho: Double): Array[Double] = {
    val AweightsExp = (A2 * weights).mapValues(x => math.exp(x))
    val diagMatrix = diag(AweightsExp.mapValues(x => x/math.pow(1+x,2)))
    val hessian = A2.t*(diagMatrix*A2) + BDM.eye[Double](A2.cols) * rho
    val invhessian =  breeze.linalg.inv(hessian)
    val diaginvHessian = (List[Double]() ++ (for(x: Int <- Range(0,A.cols)) yield {invhessian(x,x)}))
    diaginvHessian.toArray
  }

  def evalF(weights: BDV[Double],rho: Double): Double = {
    val AweightsExp = (A2 * weights).mapValues(x => math.exp(x))
    sum((AweightsExp).mapValues(x => math.log1p(x)))
  }

}
