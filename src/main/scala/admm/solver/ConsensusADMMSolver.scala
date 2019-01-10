package admm.solver

import java.io.{File, PrintWriter}

import org.apache.spark.Logging

import breeze.linalg._
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
//import org.apache.spark.mllib.linalg.LogRegressionXLocal
import admm.functions.LogRegressionXLocal
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.storage._
import org.apache.spark.broadcast._
import admm.functions._

//val f: RDF[LogRegressionXLocal],

class ConsensusADMMSolver(val f: RDF[LogRegressionXLocal],
                          val g: Function1[BDV[Double], Double] with Prox with Serializable,
                          val n: Int,
                          val absTol: Double = 10e-3,
                          val relTol: Double = 10e-3,
                          val alpha: Double,
                          var rho: Double,
                          val rhoCorrectionFactor: Double,
                          val rhoCorrectionMu: Double,
                          val lowerBoundLBFGSres: Double,
                          val solution: BDV[Double],
                          @transient val sc: SparkContext)
                          extends Serializable with Logging{


  //var filepath = "/home/gabriel/Results/Problem21_NoLasso/admmTreeA/Configuration1/test1.txt"
  //val writer = new PrintWriter(new File(filepath ))



  f.splits.cache()
  val numSplits = f.numSplits.toDouble

  /*var u_i: RDD[BDV[Double]] = f.splits.map(_ => BDV.zeros[Double](n))
  u_i.cache()*/
  var u: BDV[Double] = BDV.zeros[Double](n)

  /*var x_i: RDD[BDV[Double]] = f.splits.map(_ => BDV.zeros[Double](n))
  x_i.cache()*/
  var x: BDV[Double] = BDV.zeros[Double](n)

  /*var r_i: RDD[Double] = f.splits.map(_ => 0)
  r_i.cache()*/
  var r: Double = 0


  var z: BDV[Double] = BDV.zeros[Double](n)
  var zb: Broadcast[BDV[Double]] = sc.broadcast(z)
  //var zold = zb.value

  var iter: Int = 0


  def solve(rho: Double, maxIterations: Int = 300, evalFn: Boolean = false){
    solve(x => rho, maxIterations, evalFn)    
  }

  def solve(rho: Int => Double, maxIterations:Int, evalFn: Boolean){
    var done = false
    var iter_rho = rho(iter)
    var change = 1.0
    while(!done){
      iter += 1
      iterate()
      done = converged(evalFn) || iter >= maxIterations
    }
    //writer.close()
  }


  def iterate(){

    val toleranceLBFGS = tolLBFGS()
/*
    f.splits.foreach(sector => sector.changeU(1.0))
    f.splits.foreach(sector => sector.changeRho(1.0))
*/
    //f.splits.foreach(sector => sector.lbfgsTolerance = toleranceLBFGS)

    //println("Iter->" + iter.toString +  "\nProblem->" + solution.length + "\n")

    zb = sc.broadcast(z)
    val (unew,xnew,rnew) = f.splits.treeAggregate(BDV.zeros[Double](n),BDV.zeros[Double](n),0.0)(combinerFunction, reducerFunction)
    u = unew / numSplits
    x = xnew / numSplits
    r = Math.sqrt(rnew)
    //z = g.prox(x+u,rho)
    z = x+u;
    getNewRho()
    /*
    println("RHOval..........." + rho.toString)
    println("Uval............." + u.toString())
    println("Zval............." + z.toString())
    println("Xval............." + x.toString())
    */
    //writer.write("iter: " + iter.toString + "   zval-> " + z.toString() + "\n")

  }

  def combinerFunction: ((BDV[Double],BDV[Double],Double),LogRegressionXLocal) => (BDV[Double],BDV[Double],Double) = (Vectors,Asplit) => {
    val (unew,xnew,rnew)=Asplit.updateAll(zb.value,iter,rho)
    (Vectors._1 += unew,Vectors._2 += xnew,Vectors._3 + rnew)
  }
/*
  def combinerFunction: ((BDV[Double],BDV[Double],Double),L2NormSquared) => (BDV[Double],BDV[Double],Double) = (Vectors,Asplit) => {
    val (unew,xnew,rnew)=Asplit.updateAll(zb.value,iter,rho)
    (Vectors._1 += unew,Vectors._2 += xnew,Vectors._3 + rnew)
  }
*/
  def reducerFunction: ((BDV[Double],BDV[Double],Double),(BDV[Double],BDV[Double],Double)) => (BDV[Double],BDV[Double],Double) = (aSplit,bSplit) => (aSplit._1 + bSplit._1,aSplit._2 + bSplit._2,aSplit._3 + bSplit._3)

  def primalTolerance: Double = Math.sqrt(n)*absTol + relTol*Math.max(norm(x),norm(z))

  def dualResidual(): Double =  rho*Math.sqrt(f.numSplits)*norm(z - zb.value)

  def dualTolerance(): Double =   Math.sqrt(n)*absTol + rho*relTol*norm(u)

  def primalResidual: Double =  r

  def tolLBFGS(): Double = math.max((dualResidual + primalResidual),lowerBoundLBFGSres)

  def rhoUpdt(): Double = rho

  //def timeEvalF: Double = f.splits.map(x => x.timeEvalF).reduce(_+_)/f.numSplits().toDouble

  //def timeLBFGS: Double = f.splits.map(x => x.timeLBFGS).reduce(_+_)/f.numSplits().toDouble

  //def IterLBFGS: Double = f.splits.map(x => x.numIter  ).reduce(_+_)/f.numSplits().toDouble

  def Zval: BDV[Double] = {z}

  def converged(evalFn: Boolean): Boolean = {
    val primRes = primalResidual
    val primTol = primalTolerance
    val dualRes = dualResidual
    val dualTol = dualTolerance
    //val funVal = 0.0
    //val funVal  = fnVal(rho)
    val rhoval = rhoUpdt()
    Zval

    val converged = (primRes <= primTol) && (dualRes <= dualTol)
    /*if(evalFn){
      logInfo(f"Iteration: $iter | $funVal%.6f |  $primRes%.6f / $primTol%.6f | $dualRes%.6f / $dualTol%.6f")
    }else{
      logInfo(f"Iteration: $iter | $primRes%.6f / $primTol%.6f | $dualRes%.6f / $dualTol%.6f")
    }*/
    if(converged){ logInfo("CONVERGED") }
    return converged
  }



  def getNewRho(): Unit = {
    val primres = norm(primalResidual)
    val dualres = norm(dualResidual)
    if (primres>rhoCorrectionMu*dualres) {
      rho = rho*rhoCorrectionFactor
    }
    else if (rhoCorrectionMu*primres<dualres){
      rho = rho/rhoCorrectionFactor
    }
    //rho = math.min(rho,10000.0)
  }

  def fnVal: Double = {
    g(z) //f(z) +
  }
}
