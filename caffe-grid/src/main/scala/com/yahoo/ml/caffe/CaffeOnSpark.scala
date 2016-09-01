// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{FileReader, PrintWriter}
import java.net.InetAddress

import caffe.Caffe._

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql
import org.apache.spark.sql.types.{FloatType, StructField, StructType, ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.udf

import org.slf4j.{LoggerFactory, Logger}
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future, ExecutionContext}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.collection.mutable
import scala.collection.immutable.Map
import scala.collection.mutable.ArrayBuffer
import org.apache.spark._
import org.apache.hadoop.fs._
import org.apache.hadoop.conf._
import org.apache.hadoop.io._
import org.apache.hadoop.mapred._
import org.apache.hadoop.util._
import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.net._
import org.apache.spark.rdd._
import scala.reflect.ClassTag

object CaffeOnSpark {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def bootstrap(conf: Config, sc: SparkContext): Unit = {
    //training if specified
    val caffeSpark = new CaffeOnSpark(sc)
    if (conf.isTraining ){
//      if (conf.solverParameter.hasTestInterval && (conf.solverParameter.getTestInterval() != 0) && (conf.solverParameter.getTestIter(0) != 0)) {
//        val sourceTrain: DataSource[Any,Any] = DataSource.getSource(conf, true).asInstanceOf[DataSource[Any, Any]]
//        val sourceValidation: DataSource[Any,Any] = DataSource.getSource(conf, false).asInstanceOf[DataSource[Any, Any]]
//        caffeSpark.trainWithValidation(sourceTrain, sourceValidation)
//      } else {
        val sourceTrain: DataSource[Any,Any] = DataSource.getSource(conf, true).asInstanceOf[DataSource[Any, Any]]
        caffeSpark.train(sourceTrain)
//      }
    }

    //feature extraction
    if (conf.isFeature || conf.isTest) {
      val source = DataSource.getSource(conf, false)
      if (conf.isFeature) {
        //feature extraction
        val featureDF = caffeSpark.features(source)

        //save extracted features into the specified file
        val rdf = featureDF.write.format(source.conf.outputFormat).save(source.conf.outputPath)
      } else {
        //test
        val result = caffeSpark.test(source)

        //save test results into a local file
        val outputPath = source.conf.outputPath
        var localFilePath: String = outputPath
        if (outputPath.startsWith(FSUtils.localfsPrefix))
          localFilePath = outputPath.substring(FSUtils.localfsPrefix.length)
        else
          localFilePath = System.getProperty("user.dir") + "/test_result.tmp"
        val out: PrintWriter = new PrintWriter(localFilePath)
        result.map {
          case (name, r) => {
            out.println(name + ": " + r.mkString(","))
          }
        }
        out.close

        //upload the result file available on HDFS
        if (!outputPath.startsWith(FSUtils.localfsPrefix))
          FSUtils.CopyFileToHDFS(localFilePath, outputPath)
      }

    }
  }
  def main(args: Array[String]): Unit = {
    val sc_conf = new SparkConf()
    sc_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")

    val sc: SparkContext = new SparkContext(sc_conf)
    //Caffe-on-Spark configuration
    var conf = new Config(sc, args)
    bootstrap(conf, sc)
  }
}


/**
  * CaffeOnSpark is the main class for distributed deep learning.
  * It will launch multiple Caffe cores within Spark executors, and conduct coordinated learning from HDFS datasets.
  *
  * @param sc Spark Context
  */
class CaffeOnSpark(@transient val sc: SparkContext) extends Serializable {
  @transient private val log: Logger = LoggerFactory.getLogger(this.getClass)
  @transient val sqlContext = new sql.SQLContext(sc)
  @transient val floatarray2doubleUDF = udf((float_features: Seq[Float]) => {
    float_features(0).toDouble
  })
  @transient val floatarray2doublevectorUDF = udf((float_features: Seq[Float]) => {
    val double_features = new Array[Double](float_features.length)
    for (i <- 0 until float_features.length) double_features(i) = float_features(i)
    Vectors.dense(double_features)
  })

  private def setupTraining[T1, T2](sources: Array[DataSource[T1, T2]]): Array[String] = {
    //Phase 1: Gather RDMA addresses from executors
    val conf = sources(0).conf
    if (!conf.snapshotStateFile.isEmpty && conf.snapshotModelFile.isEmpty) {
      log.error("to resume training, please provide input model file")
      throw new IllegalStateException("input model file must be provided for incremental training")
    }

    var rank_2_addresses_n_host = sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      case rank: Int => {
        val processor = CaffeProcessor.instance[T1, T2](sources, rank)
        //announce local RDMA address
        if (conf.clusterSize > 1) {
          (rank, processor.getLocalAddress(), InetAddress.getLocalHost.getHostName)
        } else {
          (rank, new Array[String](1), InetAddress.getLocalHost.getHostName)
        }
      }
    }.collect()

    for (i <- rank_2_addresses_n_host)
      log.info("rank = " + i._1 + ", address = " + i._2.mkString(",") + ", hostname = " + i._3)
    var numExecutors: Int = sc.getExecutorMemoryStatus.size
    val numDriver: Int = if (sc.isLocal) 0 else 1
    if (!sc.isLocal && conf.clusterSize + numDriver != numExecutors) {
      log.error("Requested # of executors: " + conf.clusterSize + " actual # of executors:" + (numExecutors - numDriver) +
        ". Please try to set --conf spark.scheduler.maxRegisteredResourcesWaitingTime with a large value (default 30s)")
      throw new IllegalStateException("actual number of executors is not as expected")
    }

    //Phase 2: bcast RDMA addresses
    val rank_2_addresses = rank_2_addresses_n_host.map {
      case (rank, rdma_addr, host) => {
        if (rank == 0) log.info("rank 0:" + host)
        (rank, rdma_addr)
      }
    }
    val bcast_addresses = sc.broadcast(rank_2_addresses)

    //Phase 3: set up the processors
    val validation_blob_names = sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      case rank: Int => {
        val processor = CaffeProcessor.instance[T1, T2](rank)
        //start processor w/ the given addresses
        processor.start(bcast_addresses.value)

        if (rank==0) processor.getValidationOutputBlobNames()
        else null
      }
    }.collect()

    //return validation blob names if any
    if (validation_blob_names.length>=1) validation_blob_names.apply(0) else null
  }

  /**
    * Training with a specific data source
    * @param source input data source
    */
  def train[T1, T2](source: DataSource[T1, T2]): Unit = {
    var trainDataRDD: RDD[T1] = source.makeRDD(sc)
    if (trainDataRDD == null) {
      log.info("No training data is given")
      throw new IllegalStateException("No training data is given")
    }

    setupTraining(Array(source))
    val conf = source.conf
    //Phase 1: repartition RDD if needed
    val origin_part_count = trainDataRDD.partitions.size
    if (conf.dataPartitions % conf.clusterSize != 0) {
      throw new RuntimeException("dataPartitions % clusterSize != 0")
    }
    val desired_part_count = conf.dataPartitions
    log.info("Training dataset partition count: " + origin_part_count + " -> " + desired_part_count)
    if (origin_part_count != desired_part_count) {
      trainDataRDD = trainDataRDD.coalesce(desired_part_count, true)
    }
    trainDataRDD.mapPartitions {iter => {Iterator(iter.size)}}.collect().foreach { size =>
      if (size == 0) {
        throw new RuntimeException("empty partition")
      }
      log.info("partition_size: " + size)
    }

    //    val origin_part_count = trainDataRDD.partitions.size
//    val desired_part_count = (origin_part_count / conf.clusterSize) * conf.clusterSize
//    if (origin_part_count != desired_part_count) {
//      trainDataRDD = trainDataRDD.coalesce(desired_part_count, true)
//      log.info("Training dataset partition count: " + origin_part_count + " -> " + desired_part_count)
//    }
//    if (conf.isRddPersistent) {
//      trainDataRDD = trainDataRDD.persist(StorageLevel.DISK_ONLY)
//    }
    trainDataRDD = trainDataRDD.persist(StorageLevel.MEMORY_AND_DISK)

    //Phase 2: find the minimum size of partitions
    var minPartSize = 0
    if (conf.clusterSize > 1) {
      val sizeRDD = trainDataRDD.mapPartitionsWithIndex {
        (pId, iter) => {
          val partSize = iter.size
          // Spark decides how data partitions are distributed among executors in this step.
          // synchronize among the executors,
          // to achieve same number of partitions.
          val processor = CaffeProcessor.instance[T1, T2](pId % conf.clusterSize)
          processor.sync()
          Iterator(partSize)
        }
      }.persist()
      minPartSize = sizeRDD.min()
      log.info("Partition size: min=" + minPartSize + " max=" + sizeRDD.max())
    }

    //Phase 3: feed the processor
    var continuetrain: Boolean = true
    while (continuetrain) {
      //conduct training with dataRDD
      continuetrain = trainDataRDD.mapPartitionsWithIndex {
        (pId, iter) => {
          var res = false
          //feed training data from iterator
          val processor = CaffeProcessor.instance[T1, T2](pId % conf.clusterSize)
          if (!processor.solversFinished) {
            if (minPartSize > 0) {
              var idx = 0
              //the entire iterator needs to be consumed, otherwise GC won't be triggered
              res = iter.map { sample => {
                idx += 1
                if (idx <= minPartSize) processor.feedQueue(0, sample) else true
              }}.reduce(_ && _)
            } else {
              res = iter.map { sample => processor.feedQueue(0, sample) }.reduce(_ && _)
            }
            processor.solversFinished = !res
          }
          Iterator(res)
        }
      }.reduce(_ && _)
    }

    //Phase 4: shutdown processors
    shutdownProcessors(conf)
  }

  /**
    * Training interleaved with validation
    * @param sourceTrain input data source for training
    * @param sourceValidation input data source for validation
    * @return DataFrame of validation results
    */
  def trainWithValidation[T1, T2](sourceTrain: DataSource[T1, T2], sourceValidation: DataSource[T1, T2]): DataFrame = {
    log.info("interleave")
    var trainDataRDD: RDD[T1] = sourceTrain.makeRDD(sc)
    if (trainDataRDD == null) {
      log.info("No training data given")
      throw new IllegalStateException("No training data given")
    }

    var validationDataRDD: RDD[T1] = sourceValidation.makeRDD(sc)
    if (validationDataRDD == null) {
      log.info("No validation data given")
      throw new IllegalStateException("No validation data given")
    }

    val conf = sourceTrain.conf
    //Create train and test RDDs from parent RDD
    var continue: Boolean = true
    val no_of_records_required_per_partition_train = conf.solverParameter.getTestInterval() * sourceTrain.batchSize()  * conf.devices
    val total_records_train = trainDataRDD.count()
    log.info("total_records_train: " + total_records_train)
    log.info("no_of_records_required_per_partition_train: " + no_of_records_required_per_partition_train)
    if (total_records_train < no_of_records_required_per_partition_train * conf.clusterSize) {
      log.error("Train data is insufficient for the hyperparameters configured. Adjust the train hyperparameters or increase the training data!")
      throw new IllegalStateException("Train data is insufficient for the hyperparameters configured. Adjust the train hyperparameters or increase the training data!")
    }

    val no_of_records_required_per_partition_validation = conf.solverParameter.getTestIter(0) * sourceValidation.batchSize()
    val total_records_validation = validationDataRDD.count()
    log.info("total_records_validation: " + total_records_validation)
    log.info("no_of_records_required_per_partition_validation: " + no_of_records_required_per_partition_validation)
    if (total_records_validation < no_of_records_required_per_partition_validation * conf.clusterSize) {
      log.error("Validation data is insufficient for the hyperparameters configured. Adjust the validation hyperparameters or increase the validation data!")
      throw new IllegalStateException("Validation data is insufficient for the hyperparameters configured. Adjust the validation hyperparameters or increase the validation data!")
    }

    val validationOutputBlobNames = setupTraining(Array(sourceTrain, sourceValidation))
    implicit val rdd_class_tag : ClassTag[T1] = ClassTag.apply[T1](trainDataRDD.first.getClass)
    var zippedTrainRDD:RDD[(Long, T1)] = trainDataRDD.zipWithIndex.map{ case (e,i) => (i,e)}
    var no_of_partitions_train = total_records_train/no_of_records_required_per_partition_train
    log.info("no_of_partitions_train: " + no_of_partitions_train)
    var repartitionedTrainRDD = zippedTrainRDD.partitionBy(new FixedSizePartitioner(no_of_partitions_train.toInt+1, no_of_records_required_per_partition_train))
    if (conf.isRddPersistent) {
      repartitionedTrainRDD = repartitionedTrainRDD.persist(StorageLevel.DISK_ONLY)
    }

    var zippedValidationRDD:RDD[(Long, T1)] = validationDataRDD.zipWithIndex.map{case (e,i) => (i,e)}
    var no_of_partitions_validation = total_records_validation/no_of_records_required_per_partition_validation
    log.info("no_of_partitions_validation: " + no_of_partitions_validation)
    var repartitionedValidationRDD = zippedValidationRDD.partitionBy(new FixedSizePartitioner(no_of_partitions_validation.toInt+1, no_of_records_required_per_partition_validation))
    if (conf.isRddPersistent) {
      repartitionedValidationRDD = repartitionedValidationRDD.persist(StorageLevel.DISK_ONLY)
    }

    var current_partition_count_train = 0
    var current_partition_count_validation = 0
    var interleaveValidationRDD:RDD[(Long,T1)] = null
    val iter_train = no_of_partitions_train.toInt/conf.clusterSize.toInt
    val iter_validation = no_of_partitions_validation.toInt/conf.clusterSize.toInt
    var validation_output_rdd : RDD[Row] = null
    while(continue) {
      var interleaveTrainRDD = PartitionPruningRDD.create(repartitionedTrainRDD,
        (index => (index >= current_partition_count_train*conf.clusterSize) && (index < (current_partition_count_train+1)*conf.clusterSize))
      )
      //Proceed with the training
      continue = interleaveTrainRDD.mapPartitionsWithIndex {
        (pId, iter) => {
          var res = false
          //feed training data from iterator
          val processor = CaffeProcessor.instance[T1, T2](pId % conf.clusterSize)
          if (!processor.solversFinished) {
            res = iter.map { sample => processor.feedQueue(0, sample._2) }.reduce(_ && _)
            processor.solversFinished = !res
          }
          Iterator(res)
        }
      }.reduce(_ && _)

      if (continue) {
        //Create the interleaveValidationRDD for the required range
        interleaveValidationRDD = PartitionPruningRDD.create(repartitionedValidationRDD,
          (index => (index == current_partition_count_validation))
        )
        //Add clustersize partitions to interleaveValidationRDD where each partition is a copy of each other
        var validationRDDRef = interleaveValidationRDD
        for(j <- Range(0,conf.clusterSize-1))
          interleaveValidationRDD = interleaveValidationRDD.union(validationRDDRef)

        //Proceed with the validation
        val current_result_array : Array[Row] = interleaveValidationRDD.mapPartitionsWithIndex {
          (index, iter) => {
            //feed validation data from iterator
            val processor = CaffeProcessor.instance[T1, T2](index % conf.clusterSize)
            if (!processor.solversFinished) {
              val res = iter.map { sample => processor.feedQueue(1, sample._2)}.reduce(_ && _)
              processor.solversFinished = !res
            }

            val validation_result = if (!processor.solversFinished)
              processor.validationResultsQueue.take()
            else Iterator(null)

            if (index==0)
              validation_result
            else
              Iterator(null)
          }
        }.collect()

        continue = (current_result_array(0)!=null)
        if (continue) {
          if (validation_output_rdd == null)
            validation_output_rdd = sc.parallelize(current_result_array.toSeq, 1)
          else
            validation_output_rdd = validation_output_rdd.union(sc.parallelize(current_result_array.toSeq, 1))

          current_partition_count_train = (current_partition_count_train.toInt + 1) % iter_train
          current_partition_count_validation = (current_partition_count_validation.toInt + 1) % iter_validation
        }
      }
    }

    //shutdown processors
    shutdownProcessors(conf)

    //dataframe of validation result
    val schema = new StructType(validationOutputBlobNames.map(name => StructField(name, ArrayType(FloatType), false)))
    sqlContext.createDataFrame(validation_output_rdd, schema).persist(StorageLevel.DISK_ONLY)
  }

  /**
    * a utility function for shutting processor thread pool
    */
  private def shutdownProcessors[T1, T2](conf: Config): Unit = {
    sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      id => {
        val processor = CaffeProcessor.instance[T1, T2](id)
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
        val processor = CaffeProcessor.instance[T1, T2](Array(source), rank)
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
      sc.parallelize(0 until clusterSize, clusterSize).map { id =>
        val processor = CaffeProcessor.instance[T1, T2](id)
        processor.getValidationOutputBlobNames()
      }.collect()(0)
    val schema = new StructType(Array(StructField("SampleID", StringType, false)) ++ blobNames.map(name => StructField(name, ArrayType(FloatType), false)))
    log.info("Schema:" + schema)

    //Phase 3: feed the processors
    val featureRDD = srcDataRDD.mapPartitionsWithIndex {
      (pId, iter) => {
        val processor: CaffeProcessor[T1, T2] = CaffeProcessor.instance[T1, T2](pId % conf.clusterSize)
        val feature_iter: Iterator[Row] =
          if (processor.solversFinished)
            Iterator()
          else {
            processor.synchronized {
              processor.start(null)
              val res = iter.map { sample => processor.feedQueue(0, sample) }.reduce(_ && _)
              processor.solversFinished = !res
              processor.stopThreads()

              import scala.collection.JavaConversions._
              processor.results.iterator
            }
          }
        feature_iter
      }
    }

    //Phase 4: Create output data frame
    sqlContext.createDataFrame(featureRDD, schema).persist(StorageLevel.DISK_ONLY)
  }

}