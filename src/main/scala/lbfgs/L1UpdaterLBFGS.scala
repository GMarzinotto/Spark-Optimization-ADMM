package org.apache.spark.mllib.linalg

/**
 * Created by gabriel on 8/12/15.
 */

import breeze.linalg.{norm => brzNorm, axpy => brzAxpy, Vector => BV}

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.optimization.{Updater}


class L1UpdaterLBFGS() extends Updater {

  var store: List[Tuple3[Int,Double,BV[Double]]] = Nil
  val startTime = System.nanoTime()/1000.0


  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val thisIterStepSize = 2* stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    if(thisIterStepSize>0.0) {
      store :::= List((store.length + 1, (System.nanoTime()/1000.0 - startTime), brzWeights))
    }
    (Vectors.fromBreeze(brzWeights), 0)
  }

}
