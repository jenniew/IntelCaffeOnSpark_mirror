// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import com.yahoo.ml.caffe.tools.Binary2Sequence
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory
import org.testng.Assert._

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
    val solver_config_path = ROOT_PATH + "caffe-grid/src/test/resources/ps-integration-test/caffenet_solver.prototxt"
    val args = Array("-conf", solver_config_path,
      "-model", "file:" + ROOT_PATH + "caffe-grid/target/model.h5",
      "-imageRoot", "file:" + ROOT_PATH + "data/images",
      "-labelFile", "file:" + ROOT_PATH + "data/images/labels.txt",
      "-clusterSize", "4",
      "-devices", "1",
      "-connection", "ethernet",
      "-train"
    )

    conf = new  com.yahoo.ml.caffe.Config(sc, args)

    val seq_file_path = "file:"+ROOT_PATH+"caffe-grid/target/seq_image_files"
    val path = new Path(seq_file_path)
    val fs = path.getFileSystem(new Configuration)
    if (fs.exists(path)) fs.delete(path, true)
    val b2s = new Binary2Sequence(sc, conf)
    assertNotNull(b2s)
    b2s.makeRDD().saveAsSequenceFile(seq_file_path)

    CaffeOnSpark.bootstrap(conf, sc)
  }
}