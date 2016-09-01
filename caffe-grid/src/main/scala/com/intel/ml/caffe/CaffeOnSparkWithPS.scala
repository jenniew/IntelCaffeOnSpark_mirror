// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.intel.ml.caffe

import com.yahoo.ml.caffe.{CaffeProcessor, Config, DataSource}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.udf
import org.apache.spark.{SparkConf, SparkContext}
import org.parameterserver.client.PSClient
import org.slf4j.{Logger, LoggerFactory}

object CaffeOnSparkWithPS {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]) {
    val sc_conf = new SparkConf()
    sc_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val sc: SparkContext = new SparkContext(sc_conf)

    //Caffe-on-Spark configuration
    var conf = new Config(sc, args)
    bootstrap(conf, sc)
    // TODO: Add exit processing
    log.info("finish training")
  }

  // TODO: delete isLocal
  def bootstrap(conf: Config, sc: SparkContext, isLocal: Boolean = false): Unit = {
    //training if specified
    val caffeSpark = new CaffeOnSparkWithPS(sc)
    if (conf.isTraining) {
      val source = DataSource.getSource(conf, true)
      caffeSpark.train(source, isLocal)
    } else {
      // TODO: Add test phase implementation
      throw new RuntimeException("haven't implement yet")
    }
  }
}


/**
  * CaffeOnSpark is the main class for distributed deep learning.
  * It will launch multiple Caffe cores within Spark executors, and conduct coordinated learning from HDFS datasets.
  *
  * @param sc Spark Context
  */
class CaffeOnSparkWithPS(@transient val sc: SparkContext) extends Serializable {
  @transient private val log: Logger = LoggerFactory.getLogger(this.getClass)
  @transient val floatarray2doubleUDF = udf((float_features: Seq[Float]) => {
    float_features(0).toDouble
  })
  @transient val floatarray2doublevectorUDF = udf((float_features: Seq[Float]) => {
    val double_features = new Array[Double](float_features.length)
    for (i <- 0 until float_features.length) double_features(i) = float_features(i)
    Vectors.dense(double_features)
  })

  /**
    * Training with a specific data source
    *
    * @param source input data source
    */
  def train[T1, T2](source: DataSource[T1, T2], isLocal: Boolean): Unit = {
    // TODO: The parameter isLocal never used
    var trainDataRDD: RDD[T1] = source.makeRDD(sc)
    if (trainDataRDD == null) {
      log.info("No training data is given")
      return
    }

    val conf = source.conf

    // Phase 1: Initialize PS client context
    log.info("phase 1: Initialize PS client context")
    val psClient = new PSClient(conf.psMasterAddr)
    psClient.setContext("Caffe_On_Spark_PS_" + conf.psWeightVector)
    psClient.bspInitializeContext(conf.clusterSize)

    val workers = sc.parallelize(0 until conf.clusterSize, conf.clusterSize).cache()
    // Phase 2: new one CaffeProcessor object for each node
    log.info(s"phase 02: new one CaffeProcessor object for ${conf.clusterSize} node")
    workers.foreach { rank: Int =>
      val processor = CaffeProcessor.instance[T1, T2](source, rank)
    }

    log.info("initialize weights on each node")
    workers.foreach { rank: Int =>
      val processor = CaffeProcessor.instance[T1, T2](rank)
      processor.caffeNetList(0).initializeWeight()
    }

    //Phase 3: set up the processors
    log.info("phase 03: Initialize weights on each node and start each processor")
    // TODO: how to initialize global weight elegantly
    workers.foreach {rank: Int =>
      CaffeProcessor.instance[T1, T2](rank).start(null)
    }

    if (trainDataRDD.getNumPartitions != conf.clusterSize) {
      log.info(s"repartitioning from ${trainDataRDD.getNumPartitions} to ${conf.clusterSize}")
      trainDataRDD.repartition(conf.clusterSize)
    }

    //Phase 4: feed the processor
    log.info("phase 4: feed the processor")
    var continue: Boolean = true
    val clusterSize = conf.clusterSize
    while (continue) {
      //conduct training with dataRDD
      continue = trainDataRDD.mapPartitionsWithIndex {
        (pId, iter) => {
          var res = false
          var processor: CaffeProcessor[T1, T2] = null
          //feed training data from iterator
          processor = CaffeProcessor.instance[T1, T2](pId % clusterSize)
          if (!processor.solversFinished) {
            res = iter.map { sample => processor.feedQueue(sample) }.reduce(_ && _)
            processor.solversFinished = !res
          }
          Iterator(res)
        }
      }.reduce(_ && _)
    }

    //Phase 5: shutdown processors
    log.info("phase 5: shutdown processors")
    shutdownProcessors(conf)
  }

  /**
    * a utility function for shutting processor thread pool
    */
  private def shutdownProcessors[T1, T2](conf: Config): Unit = {
    sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      rank => {
        val processor = CaffeProcessor.instance[T1, T2](rank)
        processor.stop()
      }
    }.collect()
  }

}


