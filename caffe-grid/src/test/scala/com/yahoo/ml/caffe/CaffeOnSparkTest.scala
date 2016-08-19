// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.PrintWriter

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{ArrayType, FloatType, StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, sql}
import org.parameterserver.client.PSClient
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable
import scala.util.Random

object CaffeOnSparkTest {
//  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]) {
    val sc_conf = new SparkConf().setAppName("Caffe-on-spark-with-ps").setMaster("local[*]")
    sc_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val sc: SparkContext = new SparkContext(sc_conf)
    val ROOT_PATH = {
      val fullPath = getClass.getClassLoader.getResource("log4j.properties").getPath
      fullPath.substring(0, fullPath.indexOf("caffe-grid/"))
    }
    val solver_config_path = ROOT_PATH + "caffe-grid/src/test/resources/caffenet_solver.prototxt";
    val my_args = Array("-train",
      "-conf", solver_config_path,
      "-model", "file:"+ROOT_PATH+"caffe-grid/target/model.h5",
      "-imageRoot", "file:"+ROOT_PATH+"data/images",
      "-labelFile", "file:"+ROOT_PATH+"data/images/labels.txt",
      "-clusterSize", "2",
      "-devices", "1",
      "-connection", "ethernet",
      "-psMasterAddr", "localhost:62171", //FIXME: reset the port to your ps mini-cluster's for test
      "-psWeightVector", "w-" + Random.nextLong().toString
    )

    //Caffe-on-Spark configuration
    var conf = new Config(sc, my_args)

    //training if specified
    val caffeSpark = new CaffeOnSparkTest(sc)
    if (conf.isTraining) {
      val source = DataSource.getSource(conf, true)
      caffeSpark.train(source)
    }
  }
}


/**
  * CaffeOnSpark is the main class for distributed deep learning.
  * It will launch multiple Caffe cores within Spark executors, and conduct coordinated learning from HDFS datasets.
  *
  * @param sc Spark Context
  */
class CaffeOnSparkTest(@transient val sc: SparkContext) extends Serializable {
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
  def train[T1, T2](source: DataSource[T1, T2]): Unit = {
    var trainDataRDD: RDD[T1] = source.makeRDD(sc)
    if (trainDataRDD == null) {
      log.info("No training data is given")
      return
    }

    val conf = source.conf

    // Phase 1: Initialize PS client context
    log.info("Phase 1: Initialize PS client context")
    val psClient = new PSClient(conf.psMasterAddr)
    psClient.setContext("Caffe_On_Spark_PS_" + conf.psWeightVector)
    psClient.bspInitializeContext(conf.clusterSize)

    // Phase 2: new one CaffeProcessor object for each node
    log.info("phase 02: new one CaffeProcessor object for each node")
    sc.parallelize(0 until conf.clusterSize, conf.clusterSize).foreach({
      CaffeProcessor.instance[T1, T2](source, _)
    })

    //Phase 3: set up the processors
    log.info("phase 03: start each processor")
    sc.parallelize(0 until conf.clusterSize, conf.clusterSize).foreach(_ => {
      CaffeProcessor.instance[T1, T2]().start(null)
    })

    //Phase 4: find the minimum size of partitions
    var minPartSize = 0
    if (conf.clusterSize > 1) {
      val sizeRDD = trainDataRDD.mapPartitions {
        iter => {
          val partSize = iter.size
          Iterator(partSize)
        }
      }.persist()
      minPartSize = sizeRDD.min()
      log.info("Partition size: min=" + minPartSize + " max=" + sizeRDD.max())
    }

    //Phase 5: feed the processor
    var continue: Boolean = true
    while (continue) {
      //conduct training with dataRDD
      continue = trainDataRDD.mapPartitions {
        iter => {
          if (iter.isEmpty) {
//            log.info("encounter zero partitions")
            Iterator(true)
          } else {
            var res = false
            //feed training data from iterator
            val processor = CaffeProcessor.instance[T1, T2]()
            if (!processor.solversFinished) {
              if (minPartSize > 0) {
                res = iter.take(minPartSize).map { sample => processor.feedQueue(sample) }.reduce(_ && _)
              } else {
                res = iter.map { sample => processor.feedQueue(sample) }.reduce(_ && _)
              }
              processor.solversFinished = !res
            }
            Iterator(res)
          }
        }
      }.reduce(_ && _)
    }

    //Phase 6: shutdown processors
    shutdownProcessors(conf)
  }

  /**
    * a utility function for shutting processor thread pool
    */
  private def shutdownProcessors[T1, T2](conf: Config): Unit = {
    sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      _ => {
        val processor = CaffeProcessor.instance[T1, T2]()
        processor.stop()
      }
    }.collect()
  }

  /**
    * Test with a specific data source.
    * Test result will be saved into HDFS file per configuration.
    *
    * @param source input data source
    * @return key/value map for mean values of output layers
    */
  def test[T1, T2](source: DataSource[T1, T2]): Map[String, Seq[Double]] = {
    source.conf.isTest = true
    val testDF = features2(source)

    var result = new mutable.HashMap[String, Seq[Double]]
    // compute the mean of the columns
    testDF.columns.zipWithIndex.map {
      case (name, index) => {
        if (index > 0) {
          // first column is SampleId, ignored.
          val n: Int = testDF.take(1)(0).getSeq[Double](index).size
          val ndf = testDF.agg(new VectorMean(n)(testDF(name)))
          val r: Seq[Double] = ndf.take(1)(0).getSeq[Double](0)
          result(name) = r
        }
      }
    }

    //shutdown processors
    shutdownProcessors(source.conf)

    result.toMap
  }

  /**
    * Extract features from a specific data source.
    * Features will be saved into DataFrame per configuration.
    *
    * @param source input data source
    * @return Feature data frame
    */
  def features[T1, T2](source: DataSource[T1, T2]): DataFrame = {
    source.conf.isTest = false
    var featureDF = features2(source)

    //take action to force featureDF persisted
    featureDF.count()

    //shutdown processors
    shutdownProcessors(source.conf)

    featureDF
  }

  /**
    * Extract features from a data source
    *
    * @param source input data source
    * @return a data frame
    */
  private def features2[T1, T2](source: DataSource[T1, T2]): DataFrame = {
    val srcDataRDD = source.makeRDD(sc)
    val conf = source.conf
    val clusterSize: Int = conf.clusterSize

    //Phase 1: start Caffe processor within each executor
    val size = sc.parallelize(0 until clusterSize, clusterSize).map {
      case rank: Int => {
        // each processor has clusterSize 1 and rank 0
        val processor = CaffeProcessor.instance[T1, T2](source, rank)
      }
    }.count()
    if (size < clusterSize) {
      log.error((clusterSize - size) + "executors have failed. Please check Spark executor logs")
      throw new IllegalStateException("Executor failed at CaffeProcessor startup for test/feature extraction")
    }

    // Sanity check
    val numExecutors: Int = sc.getExecutorMemoryStatus.size
    val numDriver: Int = if (sc.isLocal) 0 else 1
    if ((size + numDriver) != sc.getExecutorMemoryStatus.size) {
      log.error("Requested # of executors: " + clusterSize + " actual # of executors:" + (numExecutors - numDriver) +
        ". Please try to set --conf spark.scheduler.maxRegisteredResourcesWaitingTime with a large value (default 30s)")
      throw new IllegalStateException("actual number of executors is not as expected")
    }

    // Phase 2 get output schema
    val blobNames = if (conf.isFeature)
      conf.features
    else // this is test mode
      sc.parallelize(0 until clusterSize, clusterSize).map { _ =>
        val processor = CaffeProcessor.instance[T1, T2]()
        processor.getTestOutputBlobNames
      }.collect()(0)
    val schema = new StructType(Array(StructField("SampleID", StringType, false)) ++ blobNames.map(name => StructField(name, ArrayType(FloatType), false)))
    log.info("Schema:" + schema)

    //Phase 3: feed the processors
    val featureRDD = srcDataRDD.mapPartitions {
      iter => {
        val processor = CaffeProcessor.instance[T1, T2]()
        if (!processor.solversFinished) {
          processor.start(null)
          val res = iter.map { sample => processor.feedQueue(sample) }.reduce(_ && _)
          processor.solversFinished = !res
          processor.stopThreads()
          import scala.collection.JavaConversions._
          processor.results.iterator
        } else {
          Iterator()
        }
      }
    }

    //Phase 4: Create output data frame
    val sqlContext = new sql.SQLContext(sc)
    sqlContext.createDataFrame(featureRDD, schema).persist(StorageLevel.DISK_ONLY)
  }

}


