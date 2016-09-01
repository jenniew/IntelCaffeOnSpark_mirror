// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory

object CaffeOnSparkTest {
  val log = LoggerFactory.getLogger(this.getClass)
  var sc: SparkContext = null
  var conf:  com.yahoo.ml.caffe.Config = null

  def main(args: Array[String]) {
    System.out.println("current dir: " + System.getProperty("user.dir"))
    // init conf
    val sc_conf = new SparkConf().setAppName("Caffe-on-spark-debug").setMaster("local[*]")
    sc_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    sc = new SparkContext(sc_conf)

    val ROOT_PATH = {
      val fullPath = getClass.getClassLoader.getResource("log4j.properties").getPath
      fullPath.substring(0, fullPath.indexOf("caffe-grid/"))
    }
    val solver_config_path = ROOT_PATH + "caffe-grid/src/test/resources/ps-integration-test/lenet_memory_solver.prototxt"
    val args = Array("-conf", solver_config_path,
      "-model", "file:" + ROOT_PATH + "caffe-grid/target/model.h5",
      "-clusterSize", "4",
      "-devices", "1",
      "-connection", "ethernet",
      "dataPartitions", "4",
      "-train"
    )

    conf = new  com.yahoo.ml.caffe.Config(sc, args)
    CaffeOnSpark.bootstrap(conf, sc)
  }
}