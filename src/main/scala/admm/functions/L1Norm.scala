package admm.functions

import breeze.linalg._
import breeze.linalg.{DenseVector => BDV}

class L1Norm(val lambda: Double, length: Int = 1024) extends Function1[BDV[Double],Double] with Prox with Serializable {
  
  def prox(x: BDV[Double], rho: Double): BDV[Double] = {
    //rintln("Umbral->" + length*lambda/rho)
    var x2 = x.map(x_i => softThreshold(x_i,length*lambda/rho))
    x2(-1) = x(-1)
    x2
  }

  def softThreshold(x: Double, thresh: Double): Double = {
    Math.max(1.0-thresh/Math.abs(x), 0.0)*x
  }

  def apply(x: BDV[Double]): Double = {
    return lambda*norm(x,1)
  }
}
