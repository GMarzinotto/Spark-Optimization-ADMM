package org.apache.spark.mllib.optimization

import breeze.linalg.norm
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.{dot,axpy,scal}
//import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.mllib.util.MLUtils


class GradientForLogRegADMM(rho: Double, UminusZ: Vector) extends Gradient {

  //def this() = this(2)

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val dataSize = data.size

    require(weights.size % dataSize == 0)
        /**
         * Here we compute the gradient for a Binary Logistic Regression.
         */
        val margin = -1.0 * dot(data, weights)
        val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
        axpy(multiplier, data, cumGradient)
		
		/**
         * Here we compute the gradient for the Euclidean Norm (rho/2)*||weights - Z + U||^2
         */
    var vb = new Array[Double](UminusZ.size)
		var i=0
		while(i<UminusZ.size){
		  vb(i) = weights(i) + UminusZ(i)
		  i = i + 1
		}
		val vectorDistGradient = Vectors.dense(vb)
		//TODO: Optimize to compute norm^2
		val normVector = norm(vectorDistGradient.toBreeze,2)
        axpy(rho, vectorDistGradient, cumGradient)
		
        if (label > 0) {
          // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
          log1pExp(margin) + ((rho*normVector*normVector)/2)
        } else {
          log1pExp(margin) - margin + ((rho*normVector*normVector)/2)
        }  
  }

  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }
}

