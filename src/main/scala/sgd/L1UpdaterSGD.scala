/**
 * Created by gabriel on 8/12/15.
 */
package org.apache.spark.mllib.linalg

import breeze.linalg.DenseVector

import scala.math._

import breeze.linalg.{norm => brzNorm, axpy => brzAxpy, Vector => BV, DenseVector => BDV}

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector=>dv, Vectors, Vector}
import org.apache.spark.mllib.optimization.{Updater}

class L1UpdaterSGD() extends Updater {

  var store: List[Tuple3[Int,Double,BV[Double]]] = Nil
  val startTime = System.nanoTime()/1000.0

    override def compute(
                          weightsOld: Vector,
                          gradient: Vector,
                          stepSize: Double,
                          iter: Int,
                          regParam: Double): (Vector, Double) = {
      val thisIterStepSize = stepSize / math.sqrt(iter)
      // Take gradient step
      val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
      brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
      // Apply proximal operator (soft thresholding)
      val shrinkageVal = regParam * thisIterStepSize
      var i = 0
      val len = brzWeights.length
      while (i < len-1) {
        val wi = brzWeights(i)
        brzWeights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
        i += 1
      }
      if(thisIterStepSize>0.0) {
        store :::= List((store.length + 1, (System.nanoTime()/1000.0 - startTime), brzWeights))
      }
      //println("Iters" + iter)
      //println("Modelo->" + brzWeights)
      (Vectors.fromBreeze(brzWeights), brzNorm(brzWeights, 1.0) * regParam)
    }

  def safeToInt(L: Long): Int = {
    val integer = L.toInt
    if(L != integer.toLong){
      println("ERRORCONVERTINGTOINT")
      System.exit(69)
    }
    integer
  }
}


