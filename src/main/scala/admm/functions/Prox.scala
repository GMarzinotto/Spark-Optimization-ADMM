package admm.functions

import breeze.linalg.{DenseVector => BDV}

trait Prox {
  
  def prox(x: BDV[Double], rho: Double): BDV[Double]


}

trait Prox2{
  def prox(x: Tuple2[BDV[Double],BDV[Double]], rho: Double): BDV[Double]
}