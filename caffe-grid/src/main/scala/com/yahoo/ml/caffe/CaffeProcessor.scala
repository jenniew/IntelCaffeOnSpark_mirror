// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.util
import java.util.concurrent.{ArrayBlockingQueue, ConcurrentHashMap, ForkJoinPool}
import java.util.ArrayList

import com.yahoo.ml.jcaffe._
import org.slf4j.{Logger, LoggerFactory}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import org.apache.spark.sql.Row

import scala.collection.immutable.Map
import scala.collection.mutable.ArrayBuffer
import java.util.concurrent.atomic.AtomicReference

private[caffe] object CaffeProcessor {
  // add here for debugging only otherwise singletons is sufficient.
  val rankToInstance = new util.HashMap[Int, CaffeProcessor[_, _]]

  def instance[T1, T2](sources: Array[DataSource[T1, T2]], rank: Int): CaffeProcessor[T1, T2] = synchronized {
    if (!rankToInstance.containsKey(rank)) {
      val myInstance = new CaffeProcessor[T1, T2](sources, rank)
      rankToInstance.put(rank, myInstance)
      return myInstance
    } else {
      throw new RuntimeException("Should not enter here, pls check the invoking sequence")
    }
  }

  def instance[T1, T2](rank: Int): CaffeProcessor[T1, T2] = synchronized {
    if (rankToInstance.containsKey(rank)) {
      return rankToInstance.get(rank).asInstanceOf[CaffeProcessor[T1, T2]]
    } else {
      assert(rankToInstance.size() == 1)
        val key = rankToInstance.keySet().iterator().next()
        return rankToInstance.get(key).asInstanceOf[CaffeProcessor[T1, T2]]
    }
  }
}

private[caffe] class QueuePair[T]  {
  val Free: ArrayBlockingQueue[T] = new ArrayBlockingQueue[T] (2)
  val Full: ArrayBlockingQueue[T] = new ArrayBlockingQueue[T] (2)
}

private[caffe] class CaffeProcessor[T1, T2](val sources: Array[DataSource[T1, T2]],
                                             val rank: Int) {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  log.info("my rank is " + rank)
  //initialize sources
  for (source <- sources) {
    if (!source.init()) {
      throw new Exception("Failed to initialize data source")
    }
  }

  val validationResultsQueue: ArrayBlockingQueue[Iterator[Row]] = new ArrayBlockingQueue[Iterator[Row]](2)
  val conf = sources(0).conf
  val solverMode: Int = sources(0).solverParameter.getSolverMode().getNumber()
  val numLocalGPUs: Int = conf.devices
  val numTotalGPUs: Int = numLocalGPUs * conf.clusterSize
  assert(sources != null)
  val poolSize = numLocalGPUs * (conf.transform_thread_per_device + 1) + conf.transform_thread_per_device
  implicit val exec = ExecutionContext.fromExecutorService(new ForkJoinPool(poolSize))
  val transformers: ArrayList[Future[_]] = new ArrayList[Future[_]]
  val solvers: ArrayList[Future[_]] = new ArrayList[Future[_]]
  var rdmaStarted = false
  var threadsStarted = false
  val objectHolder: ConcurrentHashMap[Object, Object] = new ConcurrentHashMap[Object, Object]()
  val snapshotInterval =  sources(0).solverParameter.getSnapshot()
  var STOP_MARK: (Array[String], Array[FloatBlob]) =  (Array[String](), Array())
  @volatile var results: ArrayList[Row] = new ArrayList[Row]
  @volatile var solversFinished = false
  val localModelPath : String = {
    if (sources(0).isTrain) ""
    else FSUtils.GetLocalFileName(conf.modelPath, "model.tmp")
  }

  //create a list of caffeTops
  val caffeNetList: Seq[CaffeNet] = {
    if (sources(0).isTrain) {
      // resume training if available
      val localStateFile: String = FSUtils.GetLocalFileName(conf.snapshotStateFile, "state.tmp")
      val localModelFile: String = FSUtils.GetLocalFileName(conf.snapshotModelFile, "model.tmp")
      Seq(new CaffeNet(conf.protoFile, localModelFile, localStateFile, numLocalGPUs,
          conf.clusterSize, rank, true, conf.connection, -1, 0))
    } else {
      // feature or test mode, we have numLocalGPUs caffeTops, each of them has one gpu.
      // this is to avoid create master/slave gpus where slave gpu does not do test.
      var startGPUIdx = -1
      var seq : Seq[CaffeNet] = Seq()
      for (g <- 0 until numLocalGPUs){
        val caffeNet = new CaffeNet(conf.protoFile, localModelPath, "", 1,
          1, 0, false, CaffeNet.NONE, startGPUIdx, 0)
        //in order to get GPU, we need to initialize P2P Sync 1st
        caffeNet.connect(null)
        startGPUIdx = caffeNet.deviceID(0)
        seq = seq :+ caffeNet
      }
      seq
    }
  }

  //comma separated list of RDMA address
  def getLocalAddress(): Array[String] = {
    caffeNetList(0).localAddresses
  }

  def getValidationOutputBlobNames(): Array[String] = {
    caffeNetList(0).getValidationOutputBlobNames()
  }

  //start the processor
  def start(rank2addresses: Array[(Int, Array[String])]) : Unit = {
    if (sources(0).isTrain) {
      val peer_addr = new Array[String](rank2addresses.length)
      for ((peer_rank, addrs) <- rank2addresses) {
          if (peer_rank != rank)
            peer_addr(peer_rank) = addrs(rank)
      }
      caffeNetList(0).connect(peer_addr)
    }

    //clear the source queue
    for (source <- sources)
      source.sourceQueue.clear()

    //start worker threads
    startThreads()
  }

  //start threads for transformers and solvers
  private def startThreads(): Unit = {
    //start threads only once for JVM
    if (threadsStarted) return
    results.clear
    solvers.clear
    transformers.clear


    for (g <- 0 until numLocalGPUs) {
      var queuePairTrain = new QueuePair[(Array[String], Array[FloatBlob])]()
      var queuePairSet: Array[QueuePair[(Array[String], Array[FloatBlob])]] = new Array[QueuePair[(Array[String], Array[FloatBlob])]](2)
      var queuePairValidation: QueuePair[(Array[String], Array[FloatBlob])] = null
      queuePairSet(0) = queuePairTrain
      if ((g==0) && sources.length > 1 && !sources(1).isTrain) {
        log.info("Interleave enabled")
        queuePairValidation = new QueuePair[(Array[String], Array[FloatBlob])]()
        queuePairSet(1) = queuePairValidation
      }
      if (sources(0).isTrain) {
        //start solvers w/ only rank 0 will save model
        solvers.add(Future {
          doTrain(caffeNetList(0), g, queuePairSet)
        })
        //start transformers
        for (t <- 0 until conf.transform_thread_per_device) {
          log.info("Start transformer for train in CaffeProcessor StartThreads")
          transformers.add(Future {
            doTransform(0, caffeNetList(0), g, queuePairTrain, g)
          })
        }

        if ((g == 0) && (sources.length > 1)) {
          for (t <- 0 until conf.transform_thread_per_device) {
            log.info("Start transformer for validation in CaffeProcessor StartThreads")
            transformers.add(Future {
              doTransform(1, caffeNetList(0), g, queuePairValidation, g)
            })
          }
        }
      } else {
        //start solvers for test
        solvers.add(Future {
          doFeatures(caffeNetList(g), 0, queuePairTrain)
        })
        //start transformers
        for (t <- 0 until conf.transform_thread_per_device) 
          transformers.add(Future {
            doTransform(0,caffeNetList(g), 0, queuePairTrain, g)
          })
      }
    }

      threadsStarted = true
  }

  // sync the executors
  def sync(): Unit = {
    /**
     * Multiple tasks may invoke sync() concurrently.
     * To sure that all executors are synchronized, we will execute those invocation in sequence.
     */
    synchronized {
      if (sources(0).isTrain)
        caffeNetList(0).sync
    }
  }

  //feed data to train queue
  def feedQueue(sourceId: Int, item: T1): Boolean = {
    var offer_status = false
    while (!solvers.get(0).isCompleted && !offer_status) {
      offer_status = sources(sourceId).sourceQueue.offer(item)
    }
    !solvers.get(0).isCompleted
  }

  //stop all threads
  def stopThreads(): Unit = {
    //send stop signals
    for (i <- 0 until conf.transform_thread_per_device * numLocalGPUs)
      for (j <- 0 until sources.length)
        feedQueue(j, sources(j).STOP_MARK)

    //stop transformers & solvers
    import scala.collection.JavaConversions._
    for (solver <- solvers) Await.result(solver, Duration.Inf)
    for (transformer <- transformers) {
      try {
        Await.result(transformer, Duration(1, "ms"))
      } catch {
        case e: Exception => log.warn("Some transformer threads haven't been terminated yet")
      }
    }
    threadsStarted = false
  }

  //stop all threads and the pool
  def stop(): Unit = {
    if (threadsStarted) {
      stopThreads()
    }
    exec.shutdown()
  }

  private def takeFromQueue(queue: ArrayBlockingQueue[(Array[String],
      Array[FloatBlob])], queueIdx: Int): (Array[String], Array[FloatBlob]) = {
    var tpl: (Array[String], Array[FloatBlob]) = null
    while (!solvers.get(queueIdx).isCompleted && tpl==null)
      tpl = queue.peek()

    if (solvers.get(queueIdx).isCompleted) return null
    queue.take()
  }

  private def putIntoQueue(tpl:(Array[String], Array[FloatBlob]),
      queue : ArrayBlockingQueue[(Array[String], Array[FloatBlob])],
      queueIdx: Int): Unit = {
    var status = false
    while (!solvers.get(queueIdx).isCompleted && status==false)
        status = queue.offer(tpl)
  }

  private def initialFreeQueue(sourceId: Int, queuePair: QueuePair[(Array[String], Array[FloatBlob])]): Unit = {
    val batchSize = sources(sourceId).batchSize()
    for (j <- queuePair.Free.remainingCapacity() to 1 by -1) {
      val datablob: Array[FloatBlob] = sources(sourceId).dummyDataBlobs()
      queuePair.Free.put((new Array[String](batchSize), datablob))
    }
  }

  private def doTransform(sourceId: Int, caffeNet: CaffeNet, solverIdx: Int,
                          queuePair: QueuePair[(Array[String], Array[FloatBlob])],
                          queueIdx: Int): Unit = {

    var source = sources(sourceId)
    //This will eliminate data copy by solver thread
    caffeNet.init(solverIdx)

    try {
        //This uses legacy memory data layer, will be removed in the future.
        var transformer: FloatDataTransformer = null
        if (source.transformationParameter != null) {
          transformer = new FloatDataTransformer(source.transformationParameter, source.isTrain)
        }
        var data: Array[FloatBlob] = if (transformer != null) source.dummyDataBlobs() else null
        val batchSize = source.batchSize()
        val dataHolder = source.dummyDataHolder()
        val sampleIds = new Array[String](batchSize)

        //initialize free queue now that device is set
        initialFreeQueue(sourceId, queuePair)

        while (!solvers.get(solverIdx).isCompleted && source.nextBatch(sampleIds, dataHolder)) {
          // push the data/lablels to solver thread
          val tpl = takeFromQueue(queuePair.Free, queueIdx)
          if (tpl != null) {
            // copy ids
            sampleIds.copyToArray(tpl._1)
            // processing data
            if (transformer != null) {
              val validInput: Boolean = dataHolder match {
                case (first, second) => {
                  if (first.isInstanceOf[MatVector] &&
                    second.isInstanceOf[FloatBlob]) {
                    transformer.transform(first.asInstanceOf[MatVector], data(0))
                    // copy data
                    for (idx <- 0 until data.size - 1)
                      tpl._2(idx).copyFrom(data(idx))
                    // copy label
                    tpl._2(data.size - 1).copyFrom(second.asInstanceOf[FloatBlob])
                    true
                  } else false
                }
                case _ => false
              }
              if (!validInput) {
                throw new Exception("Unsupported data type for transformer")
              }
            } else {
              dataHolder match {
                case dataBlobs: Seq[FloatBlob@unchecked] => {
                  for (vidx <- 0 until dataBlobs.size)
                    tpl._2(vidx).copyFrom(dataBlobs(vidx))
                }
                case _ => throw new Exception("Untransformed data type must be FloatBlob")
              }
            }
            putIntoQueue(tpl, queuePair.Full, queueIdx)
          }
        }
    }
    catch {
      case ex: Exception => {
        log.error("Transformer thread failed", ex)
        throw ex
      }
    } finally {
      takeFromQueue(queuePair.Free, queueIdx)
      putIntoQueue(STOP_MARK, queuePair.Full, queueIdx)
    }
  }

  private def getValidationOutputBlobs(length: Int, batchSize: Int): Seq[Array[Float]] = {
    val top_vec = caffeNetList(0).getValidationOutputBlobs(length)
    val dim_features: Seq[Int] = (0 until length).map{i => top_vec(i).count/batchSize}
    var outputArray = new Array[Array[Float]](length)
    // processing the result row by row
    for (j <- 0 until length) {
      val blob = top_vec(j)
      // If dim_feature(j) == 0, the layer does aggregation.
      val featureSize = if (dim_features(j) > 0) dim_features(j) else blob.count
      val fv = new Array[Float](featureSize)
      for (i <- 0 until batchSize)
        for (k <- 0 until featureSize)
          fv(k) += blob.cpu_data().get(k+dim_features(j) * i)/batchSize
      outputArray(j) = fv
    }
    outputArray
  }

  private def doTrain(caffeNet: CaffeNet, syncIdx: Int,
                      queuePairSet: Array[QueuePair[(Array[String], Array[FloatBlob])]]): Unit = {

    try {
      val isRootSolver: Boolean = (syncIdx == 0)
      val snapshotPrefix: String = sources(0).solverParameter.getSnapshotPrefix()
      val modelFilePrefix: String = conf.modelPath.substring(0, conf.modelPath.lastIndexOf("/") + 1)

      var tpl: (Array[String], Array[FloatBlob]) = null
      var tp2: (Array[String], Array[FloatBlob]) = null
      val initIter: Int = caffeNet.getInitIter(syncIdx)
      val maxIter: Int = caffeNet.getMaxIter(syncIdx)
      caffeNet.init(syncIdx, true)
      val validationInterval: Int = caffeNet.getTestInterval()
      val outputBlobNames: Array[String] = getValidationOutputBlobNames();
      val validation_output: ArrayBuffer[Seq[Array[Float]]] = new ArrayBuffer[Seq[Array[Float]]]()
      for (it <- initIter until maxIter if (tpl != STOP_MARK)) {
        var validationTime: Boolean = sources.length > 1 && (validationInterval > 0) && (it % validationInterval == 0) && (it > 0) &&  isRootSolver 
        if (validationTime) {
          validation_output.clear()
          for (testit <- 0 until caffeNet.getTestIter(0)) {
            tp2 = queuePairSet(1).Full.take
            caffeNet.validation(tp2._2)
            validation_output += getValidationOutputBlobs(outputBlobNames.length, sources(1).batchSize)
            queuePairSet(1).Free.put(tp2)
          }
          validationResultsQueue.put(validation_output.map(e => Row.fromSeq(e)).toIterator)
          caffeNet.aggregateValidationOutputs()
        }
      
        tpl = queuePairSet(0).Full.take
        if (tpl == STOP_MARK)  {
          queuePairSet(0).Free.put(tpl)
        } else {
          var rs : Boolean = false
          rs = caffeNet.train(syncIdx, tpl._2)

          if (!rs) {
            log.warn("Failed at training at iteration "+it)
          }
          queuePairSet(0).Free.put(tpl)

          if ((rank == 0) && isRootSolver && (snapshotInterval > 0) && ((it + 1) % snapshotInterval == 0)) {
            log.info("Snapshot saving into files at iteration #" + (it + 1))
            val modelFilename: String = modelFilePrefix + snapshotPrefix + "_iter_" + (it + 1)
            FSUtils.GenModelOrState(caffeNet, modelFilename, true)
          }
        }
      }

      if ((sources.length > 1) && (validationInterval > 0) && isRootSolver)
        validationResultsQueue.put(Iterator(null))
      if ((rank == 0) && isRootSolver) {
        log.info("Model saving into file at the end of training:" + conf.modelPath)
        FSUtils.GenModelOrState(caffeNet, conf.modelPath, false)
      }
    } catch {
      case ex: Exception => {
        log.error("Train solver exception", ex)
      }
    }
  }

  private def doFeatures(caffeNet: CaffeNet, syncIdx: Int,
                     queuePair: QueuePair[(Array[String], Array[FloatBlob])]): Unit = {
    try {
      var blobNames = conf.features
      if (conf.isTest)
        blobNames = getValidationOutputBlobNames()
      var act_iter: Int = 0
      var tpl: (Array[String], Array[FloatBlob]) = null
      val max_iter: Int = caffeNet.getMaxIter(syncIdx)
      val batchSize = sources(0).batchSize()
      val bl = blobNames.length
      caffeNet.init(syncIdx, true)
      while (act_iter < max_iter && tpl != STOP_MARK) {
        tpl = queuePair.Full.take
        if (tpl == STOP_MARK) {
          queuePair.Free.put(tpl)
        } else {
          val top_vec = caffeNet.predict(syncIdx, tpl._2, blobNames)
          val dim_features: Seq[Int] = (0 until bl).map{i => top_vec(i).count/batchSize}
          for (i <- 0 until batchSize) {
            // processing the result row by row
            // first item is the SampleID
            var result: Array[_] = Array(tpl._1(i))
            for (j <- 0 until bl) {
              val blob = top_vec(j)
              val offset:Int = dim_features(j) * i
              // If dim_feature(j) == 0, the layer does aggregation.
              // We repeat the feature values for individual samples in the batch.
              // To avoid this, batch size = 1 is recommended.
              val featureSize = if (dim_features(j) > 0) dim_features(j) else blob.count
              val fv = new Array[Float](featureSize)
              for (k <- 0 until featureSize) {
                fv(k) = blob.cpu_data().get(k + offset)
              }
              result = result :+ fv
            }
            results synchronized {
              results.add(Row.fromSeq(result))
            }
          }
          queuePair.Free.put(tpl)
          act_iter += 1
        }
      }
    } catch {
      case ex: Exception => {
        log.error("Test/Feature solver exception", ex)
        throw ex
      }
    }
  }
}
