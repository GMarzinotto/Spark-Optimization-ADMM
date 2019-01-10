package admm.solver

import java.io.{File, PrintWriter}

import breeze.linalg.{DenseVector => BDV, norm}

import collection.mutable
import scala.collection.mutable.MutableList

trait Instrumentation extends ConsensusADMMSolver {
  var iterData: MutableList[mutable.Map[String,Any]] = MutableList()
  var currData: mutable.Map[String,Any] = mutable.Map[String,Any]()

  override def iterate(){
    currData("rho") = rho
    currData("iter") = iter
    time("iterTime", { super.iterate() })
  }

  override def converged(evalFn: Boolean):Boolean = {
    val res = super.converged(evalFn)
    iterData :+= currData
    currData = mutable.Map[String,Any]()
    res
  }

  override def primalTolerance: Double = {
    store("primalTolerance",{ super.primalTolerance })
  }

  override def primalResidual: Double = {
    store("primalResidual",{ super.primalResidual })
  }

  override def dualTolerance: Double = {
    store("dualTolerance",{ super.dualTolerance })
  }

  override def dualResidual: Double = {
     store("dualResidual",{ super.dualResidual })
  }

  override def fnVal: Double = {
    store("fnVal",{ time("evalFnTime",{ super.fnVal }) })
  }


  override def rhoUpdt: Double = {
    store("rhoVal",{ super.rhoUpdt })
  }

  override def tolLBFGS: Double = {
    store("tolLBFGS",{ super.tolLBFGS })
  }

 /*
  override def timeEvalF: Double = {
    store("timeEvalF",{ super.timeEvalF })
  }

  override def timeLBFGS: Double = {
    store("timeLBFGS",{ super.timeLBFGS })
  }

  override def IterLBFGS: Double = {
    store("IterLBFGS",{ super.IterLBFGS })
  }
*/

  override def Zval: BDV[Double] = {
    store("Zval",{ super.Zval })
  }

  def time[R](key: String, block: => R): R = {
    val t0 = System.nanoTime()
    val result = block 
    val t1 = System.nanoTime()
    currData(key) = (t1 - t0)/1000.0
    result
  }

  def store[R <: Any](key: String, block: => R): R = {
    val result = block
    currData(key) = result
    result
  }

  def printData = {
    //println("iteration,dualResidual,dualTolerance,primalResidual,primalTolerance,iterTime,FunctionValue,RhoValue,tolLBFGS,distanceToSolution,timeEvalF,timeLBFGS,IterLBFGS,MemoryUse,Zval")
    println("iteration,iterTime,RhoValue,dualResidual,dualTolerance,primalResidual,primalTolerance,tolLBFGS,Zval")
    iterData.map(printIterData)
  }

  def printFile(DestinationPath: String,solveTime: Double, TrueModel: BDV[Double], FinalResult: BDV[Double]) = {
    val writer = new PrintWriter(new File(DestinationPath ))
    writer.write("Initialization Time: " + solveTime + "\n")
    writer.write("Expected Result (If NoLasso): " + TrueModel.toString() + "\n")
    writer.write("Final Result:   " + FinalResult.toString() + "\n")
    writer.write("iteration,iterTime,RhoValue,dualResidual,dualTolerance,primalResidual,primalTolerance,tolLBFGS,Zval\n")

    iterData.map(x => printIterDataToFile(x,writer))
    writer.close()
  }

  def printIterData(data: mutable.Map[String,Any]) = {
    val str = data("iter") +"," +
        data("iterTime") + "," + data("rhoVal") + "," +
        data("dualResidual") + "," + data("dualTolerance") + "," +
        data("primalResidual") + "," + data("primalTolerance") + "," +
        data("tolLBFGS") + "," + data("Zval")
    println(str)
  }

  def printIterDataToFile(data: mutable.Map[String,Any],writer: PrintWriter) = {
   val str = data("iter") +"," +
      data("iterTime") + "," + data("rhoVal") + "," +
      data("dualResidual") + "," + data("dualTolerance") + "," +
      data("primalResidual") + "," + data("primalTolerance") + "," +
      data("tolLBFGS")  + ","  + data("Zval").toString.stripPrefix("DenseVector(").stripSuffix(")").trim + "\n"
    writer.write(str)
  }
}



trait Instrumentation2 extends ConsensusADMMSolver {
  var iterData: MutableList[mutable.Map[String,Any]] = MutableList()
  var currData: mutable.Map[String,Any] = mutable.Map[String,Any]()

  override def iterate(){
    currData("rho") = rho
    currData("iter") = iter
    time("iterTime", { super.iterate() })
  }

  override def converged(evalFn: Boolean):Boolean = {
    val res = super.converged(evalFn)
    iterData :+= currData
    currData = mutable.Map[String,Any]()
    res
  }

  override def Zval: BDV[Double] = {
    store("Zval",{ super.Zval })
  }

  def time[R](key: String, block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    currData(key) = (t1 - t0)/1000.0
    result
  }

  def store[R <: Any](key: String, block: => R): R = {
    val result = block
    currData(key) = result
    result
  }

  def printData = {
    //println("iteration,dualResidual,dualTolerance,primalResidual,primalTolerance,iterTime,FunctionValue,RhoValue,tolLBFGS,distanceToSolution,timeEvalF,timeLBFGS,IterLBFGS,MemoryUse,Zval")
    println("iteration,iterTime,Zval")
    iterData.map(printIterData)
  }

  def printFile(DestinationPath: String,solveTime: Double, TrueModel: BDV[Double], FinalResult: BDV[Double]) = {
    val writer = new PrintWriter(new File(DestinationPath ))
    writer.write("Initialization Time: " + solveTime + "\n")
    writer.write("Expected Result (If NoLasso): " + TrueModel.toString() + "\n")
    writer.write("Final Result:   " + FinalResult.toString() + "\n")
    writer.write("iteration,iterTime,Zval\n")

    iterData.map(x => printIterDataToFile(x,writer))
    writer.close()
  }

  def printIterData(data: mutable.Map[String,Any]) = {
    val str = data("iter") +"," +
      data("iterTime") + "," + data("Zval")
    println(str)
  }

  def printIterDataToFile(data: mutable.Map[String,Any],writer: PrintWriter) = {
    val str = data("iter") +"," +
      data("iterTime") + ","  + data("Zval").toString.stripPrefix("DenseVector(").stripSuffix(")").trim + "\n"
    writer.write(str)
  }
}