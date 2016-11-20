Spark Scala Machine Learning Examples
=====================================

This project aims at demonstrating how to build a [Spark 2.0](https://spark.apache.org/releases/spark-release-2-0-0.html) application with [Scala](http://www.scala-lang.org/) for solving Machine Learning problems, packaged with [SBT](http://www.scala-sbt.org/) and ready to be run locally or on any cloud platform such as [AWS Elastic MapReduce (EMR)](https://aws.amazon.com/emr/).  

Each class/object in the package can be run as an individual application, as described below.  

### AllstateClaimsSeverityGBTRegressor and AllstateClaimsSeverityRandomForestRegressor

[Kaggle](https://www.kaggle.com) and [Allstate](https://www.allstate.com/) launched a Machine Learning [competition](https://www.kaggle.com/c/allstate-claims-severity) on predicting *how severe is an insurance claim*. These two Scala scripts obtain the training and test input datasets, from local or [S3](https://aws.amazon.com/s3/details/) environment, and train [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) and [Random Forest](https://en.wikipedia.org/wiki/Random_forest) models over it, respectively.
The objective is to demonstrate the use of [Spark 2.0](https://spark.apache.org/releases/spark-release-2-0-0.html) Machine Learning pipelines with [Scala language](http://www.scala-lang.org/), [S3](https://aws.amazon.com/s3/details/) integration and some general good practices for building Machine Learning models. In order to keep this main objective, more sophisticated techniques (such as a thorough exploratory data analysis and feature engineering) are intentionally omitted.

#### Flow of Execution and Overall Learnings

Although not so labored in terms of Machine Learning techniques, these scripts provide many important learnings for building ML applications with Spark 2.0, Scala, SBT and finally running it. Some learnings are detailed as follows:  
 
* Both scripts provide a sophisticated command line interface with [scopt](https://github.com/scopt/scopt), through which the runtime can be configured with specific named parameters. It is detailed in the section [Running the Scripts Locally](#running-the-scripts-locally). You must add this to your `build.sbt` file:

    ```scala
    libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"
    ```
    
    And your script code will include something like this: 
    
    ```scala
    val parser = new OptionParser[Params]("AllstateClaimsSeverityRandomForestRegressor") {
      head("AllstateClaimsSeverityRandomForestRegressor", "1.0")
    
      opt[String]("s3AccessKey").required().action((x, c) =>
        c.copy(s3AccessKey = x)).text("The access key is for S3")
    
      opt[String]("s3SecretKey").required().action((x, c) =>
        c.copy(s3SecretKey = x)).text("The secret key is for S3")
    ...
    ```
        
    ```scala
    parser.parse(args, Params()) match {
      case Some(params) =>
        process(params)
      case None =>
        throw new IllegalArgumentException("One or more parameters are invalid or missing")
    }
    ```
    
* In order for SBT to package a jar file containing this and other third-part libraries, you need to use the command `sbt assembly` instead of `sbt package`. For such, it is needed to use [sbt-assembly](https://github.com/sbt/sbt-assembly) and configure your project accordingly by creating a file `project/assembly.sbt` with the following content:
 
    ```scala
    resolvers += Resolver.url("artifactory", url("http://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases"))(Resolver.ivyStylePatterns)
    
    addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.3")
    ```
    
* The method `process` is called with a *case class* instance which encapsulates the parameters provided at the command line.
    
    ```scala
    case class Params(s3AccessKey: String = "", s3SecretKey: String = "",
                      trainInput: String = "", testInput: String = "",
                      outputFile: String = "",
                      algoNumTrees: Seq[Int] = Seq(3),
                      algoMaxDepth: Seq[Int] = Seq(4),
                      algoMaxBins: Seq[Int] = Seq(32),
                      numFolds: Int = 10,
                      trainSample: Double = 1.0,
                      testSample: Double = 1.0)
    ```
    
    ```scala
    def process(params: Params) {
       ...
    ```

* *SparkSession.builder* is used for building a *Spark session*. It was introduced in Spark 2.0 and is recommended to be used in place of the old *SparkConf* and *SparkContext*. [This link](https://databricks.com/blog/2016/08/15/how-to-use-sparksession-in-apache-spark-2-0.html) provides a good description of this new strategy and the equivalence with the old one.
    
    ```scala
    val sparkSession = SparkSession.builder.
      appName("AllstateClaimsSeverityRandomForestRegressor")
      .getOrCreate()
    ```

* The access to S3 is configured with **s3a** support, which compared to the predecessor **s3n** improves the support to large files (no more 5GB limit) and provides higher performance. For more information on this, check [this](https://wiki.apache.org/hadoop/AmazonS3), [this](https://aws.amazon.com/premiumsupport/knowledge-center/emr-file-system-s3/) and [this](http://stackoverflow.com/questions/30385981/how-to-access-s3a-files-from-apache-spark) links.

    ```scala
    sparkSession.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    sparkSession.conf.set("spark.hadoop.fs.s3a.access.key", params.s3AccessKey)
    sparkSession.conf.set("spark.hadoop.fs.s3a.secret.key", params.s3SecretKey)
    ```

* Besides using the new **sparkSession.read.csv** method, the reading process also includes important settings: It is set to read the header of the CSV file, which is directly applied to the columns' names of the dataframe created; and **inferSchema** property is set to *true*. Without the **inferSchema** configuration, the float values would be considered as *strings* which would later cause the **VectorAssembler** to raise an ugly error: `java.lang.IllegalArgumentException: Data type StringType is not supported`. Finally, both raw dataframes are *cached* since they are again used later in the code for *fitting* the **StringIndexer** transformations and it wouldn't be good to read the CSV files from the filesystem or S3 once again.

    ```scala
    val trainInput = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(params.trainInput)
      .cache
    
    val testInput = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(params.testInput)
      .cache
    ```
  
* The column "loss" is renamed to "label". For some reason, even after using the *setLabelCol* on the regression model, it still looks for a column called "label", raising an ugly error: `org.apache.spark.sql.AnalysisException: cannot resolve 'label' given input columns`. It may be hardcoded somewhere in Spark's  source code.
 
* The content of *train.csv* is split into *training* and *validation* data, 70% and 30%, respectively. The content of "test.csv" is reserved for building the final CSV file for submission on Kaggle. Both original dataframes are sampled according to command line parameters, which is particularly useful for running fast executions in your local machine;
  
    ```scala
    val data = trainInput.withColumnRenamed("loss", "label")
      .sample(false, params.trainSample)
    
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, validationData) = (splits(0), splits(1))
    
    trainingData.cache
    validationData.cache
    
    val testData = testInput.sample(false, params.testSample).cache
    ```
  
* By using a custom function *isCateg* the column names are filtered and a [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#stringindexer) is created for each categorical column, aimed at creating a new numerical column according to the custom function *categNewCol*. Note: It is a weak feature engineering, since it is wrong for a learning model to assume that the categories have an order among them (one is greater or less than the other). Whenever categories are confirmed to be unordered, it is better to use some other technique such as [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder), which yields a different new column for each category holding a boolean (0/1) value;

    ```scala
    def isCateg(c: String): Boolean = c.startsWith("cat")
    def categNewCol(c: String): String = if (isCateg(c)) s"idx_${c}" else c
    
    val stringIndexerStages = trainingData.columns.filter(isCateg)
      .map(c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(categNewCol(c))
        .fit(trainInput.select(c).union(testInput.select(c))))
    ```
  
* There are some very important aspects to be considered when building a feature transformation such as StringIndexer or OneHotEncoder. Such transformations need to be *fitted* before being included in the pipeline and the *fit* process needs to be done over a dataset that contains all possible categories. For instance, if you fit a StringIndexer over the training dataset and afterwards, when the pipeline is used to predict an outcome over another dataset (validation, test, etc.), it faces some unseen category, then it will fail and raise the error: `org.apache.spark.SparkException: Failed to execute user defined function($anonfun$4: (string) => double) ... Caused by: org.apache.spark.SparkException: Unseen label: XYZ ... at org.apache.spark.ml.feature.StringIndexerModel`. This is the reason why the scripts' code fits the StringIndexer transformations over a union of original data from `train.csv` and `test.csv`, bypassing the sampling and split parts.
 
* After the sequence of StringIndexer transformations, the next transformation in the pipeline is the [VectorAssembler](http://spark.apache.org/docs/latest/ml-features.html#vectorassembler), which groups a set of columns into a new "features" column to be considered by the regression model. The filter for only feature columns is performed with the custom function *onlyFeatureCols*. Additionally, the custom function *removeTooManyCategs* is used to filter out some few columns which contain a number of distinct categories much higher than the supported by the default parameter *maxBins* (for RandomForest). In a seriously competitive scenario, it would be better to perform some exploratory analysis to understand these features,  their impact on the outcome variable and which feature engineering techniques could be applied.

    ```scala
    def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$|112$|113$|116$)")
    
    def onlyFeatureCols(c: String): Boolean = !(c matches "id|label")
    
    val featureCols = trainingData.columns
      .filter(removeTooManyCategs)
      .filter(onlyFeatureCols)
      .map(categNewCol)
    
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    ```
  
* The very last stage in the pipeline is the regression model, which in these scripts is [GBTRegressor](http://spark.apache.org/docs/2.0.1/api/java/org/apache/spark/ml/regression/GBTRegressor.html) and [RandomForestRegressor](http://spark.apache.org/docs/2.0.1/api/java/org/apache/spark/ml/regression/RandomForestRegressor.html).

    ```scala
    val algo = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("label")
    
    val pipeline = new Pipeline().setStages((stringIndexerStages :+ assembler) :+ algo)
    ```
  
* It is interesting to run the pipeline a set of times with different *hyperparameters* for the transformations and the learning algorithm in order to find the combination that best fits the data (see [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)). It is also important to evaluate each combination against a separated slice of the data (see [K-fold Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))). For accomplishing such objectives, a [CrossValidator](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/tuning/CrossValidator.html) is used in conjunction with a [ParamGridBuilder](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/tuning/ParamGridBuilder.html) (more documentation on (this link)[http://spark.apache.org/docs/latest/ml-tuning.html]) queueing  executions with distinct combinations of *hyperparameters* according to which was parametrized in the command line.

    ```scala
    val paramGrid = new ParamGridBuilder()
      .addGrid(algo.numTrees, params.algoNumTrees)
      .addGrid(algo.maxDepth, params.algoMaxDepth)
      .addGrid(algo.maxBins, params.algoMaxBins)
      .build()
    
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(params.numFolds)
    
    val cvModel = cv.fit(trainingData)
    ```
  
* Note: As observed by [this post](https://databricks.com/blog/2015/01/21/random-forests-and-boosting-in-mllib.html) the Random Forest model is much faster than GBT on Spark. I experienced an execution about 20 times slower with GBT compared to Random Forest with equivalent *hyperparameters*.

* With an instance of [CrossValidatorModel](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/tuning/CrossValidatorModel.html) already trained, it is time for evaluating the model over the whole training and the validation datasets. From the result of predictions it is possible to easily obtain evaluation metrics with [RegressionMetrics](http://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/RegressionMetrics.html). Additionally, the instance of the best model can be obtained, providing thus access to some other interesting attributes, such as *featureImportances*.

    ```scala
    val trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
    
    val validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
    
    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)
    
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val featureImportances = bestModel.stages.last.asInstanceOf[RandomForestRegressionModel].featureImportances.toArray
    ```
  
* Finally, the model can be used to predict the answer for the *test* dataset and save a csv file ready to be submitted on Kaggle! Again, Spark 2.0 simplifies the process. The function `coalesce` gathers all partitions into 1 only, thus saving a single output file (not many). 
 
    ```scala
    cvModel.transform(testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save(params.outputFile)
    ```
  

#### Running the Scripts Locally

Assuming you have your local environment all set up with Java 8 or higher, Scala 2.11.x and Spark 2.0, you can run the desired script (here, AllstateClaimsSeverityRandomForestRegressor) with the following command structure:

    ```
    spark-submit --class com.adornes.spark.kaggle.AllstateClaimsSeverityRandomForestRegressor the_jar_file.jar --s3AccessKey YOUR_AWS_ACCESS_KEY_HERE --s3SecretKey YOUR_AWS_SECRET_KEY_HERE --trainInput "file:///path/to/the/train.csv" --testInput "file:///path/to/the/test.csv" --outputFile  "file:///path/to/any/name/for/submission.csv" --algoNumTrees 3 --algoMaxDepth 3 --algoMaxBins 32 --numFolds 5 --trainSample 0.01 --testSample 0.01
    ```

As previously mentioned, [scopt](https://github.com/scopt/scopt) is the tool that enables the nice names for parameters at command line. If you type something wrong, it will output the sample usage as follows:

    ```
    AllstateClaimsSeverityRandomForestRegressor 1.0
    Usage: AllstateClaimsSeverityRandomForestRegressor [options]
    
      --s3AccessKey <value>    The access key for S3
      --s3SecretKey <value>    The secret key for S3
      --trainInput <file>      Path to file/directory for training data
      --testInput <file>       Path to file/directory for test data
      --outputFile <file>      Path to output file
      --algoNumTrees <n1[,n2,n3...]>
                               One or more options for number of trees for RandomForest model. Default: 3
      --algoMaxDepth <n1[,n2,n3...]>
                               One or more values for depth limit
      --algoMaxBins <n1[,n2,n3...]>
                               One or more values for depth limit
      --numFolds <value>       Number of folds for K-fold Cross Validation
      --trainSample <value>    Sample fraction from 0.0 to 1.0 for train data
      --testSample <value>     Sample fraction from 0.0 to 1.0 for test data
    ```

#### Running the Scripts on AWS Elastic MapReduce (EMR)

**EMR** plays the role of abstracting most of the background setup for a cluster with Spark/Hadoop ecosystems. You can actually build as many clusters as you want (and can afford). By the way, the cost for EC2 instances used with EMR is considerably reduced (it is detailed [here](https://aws.amazon.com/emr/pricing)).
 
 ##### Creating the cluster
 
 Although considerably abstracting the cluster configuration, EMR allows the user to customize almost any of the background details through the *advanced* options of the steps of creating a cluster. For instance, for these Spark scripts, you'll need to customize the Java version, according to [this link](http://docs.aws.amazon.com/ElasticMapReduce/latest/ReleaseGuide/emr-configure-apps.html#configuring-java8). Besides that, everything is created using the options provided. So, going step by step, log in to your AWS console, in the *Services* tab look for *EMR*, select to create a cluster, choose *Go to advanced options* on the top of the screen and fill the options as follows: 
 
 * **Vendor** - Leave it as *Amazon*
 
 * **Release** - Choose *emr-5.1.0*. Select *Hadoop* and *Spark*. I'd also recommend you to select *Zeppelin* (for working with notebooks) and *Ganglia* (for detailed monitoring of your cluster).
 
 * **Release** - Choose *emr-5.1.0*
 
 * **Edit software settings (optional)** - Ensure the option *Enter configuration* is selected and copy here the configurations of [the aforementioned link](http://docs.aws.amazon.com/ElasticMapReduce/latest/ReleaseGuide/emr-configure-apps.html#configuring-java8)
   
 * **Release** - Choose *emr-5.1.0*
   
 * **Add steps** - You don't need to do it at this moment. I prefer to do it later, after your cluster is started and ready for processing stuff. Click Next for *Hardware* settings.
 
 * **Hardware** - You can leave it as default (and can also resize it later) but maybe 2 core instances can be increased to 4 or more. Don't forget that your choice will have costs. Click Next for *General Cluster Settings*.
 
 * **Cluster name** - Give some name to your cluster. Feel free to leave all other options with the default values. Click Next for *Security*.
 
 * **EC2 Key Pair** - It is useful if want to log into your EC2 instances via ssh. You can either create a Key Pair or choose some existent if you already have one. Leave the remaining options with the default values and click on *Create Cluster*.
 
 Now you'll have an overview of your cluster's basic data, including the state of your instances. When they indicate to be ready for processing steps, go to the **Steps** tab, click on **Add step** and fill the options as follows:
 
 * **Step type** - Select *Spark application*
  
 * **Application location** - Navigate through your S3 buckets and select the jar file there. You'll need to have already uploaded it to S3.
 
 * **Spark-submit options** - Type here `--class com.adornes.spark.kaggle.AllstateClaimsSeverityRandomForestRegressor` indicating the class that holds the code that you want to run.
 
 * **Arguments** - Here you type the rest of the command arguments as demonstrated before, but this time indicating S3 paths as follows:
 
 ```
 --s3AccessKey YOUR_AWS_ACCESS_KEY_HERE --s3SecretKey YOUR_AWS_SECRET_KEY_HERE 
 --trainInput "s3:/path/to/the/train.csv" --testInput "s3:/path/to/the/test.csv" 
 --outputFile  "s3:/path/to/any/name/for/submission.csv" 
 --algoNumTrees 3 --algoMaxDepth 3 --algoMaxBins 32 --numFolds 5 
 --trainSample 0.01 --testSample 0.01
 ```

That's it! In the list of steps you will see your step running and will also have access to system logs. Detailed logs will be saved to the path defined in your cluster configuration. Additionally, EMR allows the user to clone both steps and clusters, being thus not required to type everything again.


### Corrections/Suggestions or just a Hello!

Don't hesitate to contact me directly or create *pull requests* here if you have any correction or suggestion for the code or for this documentation! Thanks! 

[Github](https://www.github.com/adornes)
[Twitter](https://twitter.com/daniel_adornes)
[LinkedIn](https://www.linkedin.com/in/adornes)
