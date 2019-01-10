package org.apache.spark.mllib.linalg

import breeze.optimize.DiffFunction
import breeze.linalg.{Vector => BV, DenseVector => BDV, DenseMatrix => BDM, sum, *}
import org.apache.spark.mllib.optimization.{Gradient, Updater}

/**
 * Created by gabriel on 9/8/15.
 */
class CostFuncX(data: BDM[Double], var Z: BDV[Double], var U: BDV[Double], var rho: Double) extends DiffFunction[BDV[Double]] with Serializable{

  val dataParsed = (data(*,::)).map(v => if(v(-1)==1) -v else { v(-1)=1; v })
  val blocksize = data.rows.toDouble


  override def calculate(weights: BDV[Double]): (Double,BDV[Double]) = {
    var vectorsup = weights + U - Z
    val normSqrdVector = sum(vectorsup.mapValues(v => v * v))
    val AweightsExp = (dataParsed * weights).mapValues(x => math.exp(x))
    val costf = (sum((AweightsExp).mapValues(x => math.log1p(x))) + (rho / 2) * normSqrdVector)
    vectorsup :*= rho
    val gradf = (vectorsup + (dataParsed.t * (AweightsExp).mapValues(x => x / (1 + x))))
    (costf, gradf)
  }

  def updateU(newU: BDV[Double]): Unit = {U = newU}

  def updateZ(newZ: BDV[Double]): Unit = {Z = newZ}

  def updateRho(newRho: Double): Unit = {rho = newRho}

}
