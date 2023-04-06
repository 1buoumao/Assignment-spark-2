import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {
  def fillblanks(DF: DataFrame):DataFrame = {
    DF.na.fill(Map("PassengerId"-> 0, "Pclass" -> 3,"Name"-> "unknown","sex"-> "unknown"
      ,"age" -> 23,"SibSp" -> 1, "Parch" -> 0, "Ticket" -> 0, "Fare" -> 0, "Cabin" -> "unknown", "Embarked" -> "unknown"))
  }
  def fillNullValues(DF: DataFrame): DataFrame = {
    DF.na.fill(Map("PassengerId" -> 0, "Pclass" -> 3,  "sex" -> "unknown", "age" -> 23,  "Fare" -> 0,
      "Embarked" -> "unknown", "Title" -> 0, "FamilySize" -> 2, "IsAlone" -> 1
    ))
  }

  def main(args: Array[String]): Unit = {
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.SparkSession

    val spark = SparkSession.builder()
      .appName("Titanic Survival Prediction")
      .master("local[*]")
      .getOrCreate()


    val trainDF: DataFrame = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/train.csv")

    val testDF: DataFrame = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/test.csv")

//    trainDF.describe().show()
//    val filledTrainDF = trainDF.na.fill(0)
//    val filledTestDF = testDF.na.fill(0)
    val filledTrainDF = fillblanks(trainDF)
    val filledTestDF = fillblanks(testDF)


    //    trainDF.describe().show()
    //    trainDF.groupBy("Survived").count().show()
    val trainWithTitleDF = filledTrainDF.withColumn("Title", regexp_extract(col("Name"), "(?<=, ).*?(?=[.])", 0))
    val testWithTitleDF = filledTestDF.withColumn("Title", regexp_extract(col("Name"), "(?<=, ).*?(?=[.])", 0))

    val trainWithFamilySizeDF = trainWithTitleDF.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    val testWithFamilySizeDF = testWithTitleDF.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)

    val trainWithIsAloneDF = trainWithFamilySizeDF.withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))
    val testWithIsAloneDF = testWithFamilySizeDF.withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))

    val trainCleanDF = fillNullValues(trainWithIsAloneDF.drop("Name", "Ticket", "Cabin", "SibSp", "Parch"))
    val testCleanDF = fillNullValues(testWithIsAloneDF.drop("Name", "Ticket", "Cabin", "SibSp", "Parch"))
//    val trainCleanDF = trainWithIsAloneDF
//    val testCleanDF = testWithIsAloneDF

//    trainCleanDF.describe().show()
//    testCleanDF.describe().show()

    //    println(trainCleanDF.getClass.getSimpleName)
    val categoricalCols = Array("Sex", "Embarked", "Title")

    val stringIndexer = new StringIndexer()
      .setInputCols(categoricalCols)
      .setOutputCols(categoricalCols.map(colName => s"${colName}_idx"))
      .setHandleInvalid("skip")

    val featureCols = Array("Pclass", "Age", "Fare", "FamilySize", "IsAlone") ++ categoricalCols.map(colName => s"${colName}_idx")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.01)
      .setElasticNetParam(0.8)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    println("_"*100)
    // train
    val indexerModel = stringIndexer.fit(trainCleanDF)
    val trainIndexedDF = indexerModel.transform(trainCleanDF)
    trainIndexedDF.describe().show()
    val assembledTrainDF = assembler.transform(trainIndexedDF)
    val model = lr.fit(assembledTrainDF)
    val trainPredictionsDF = model.transform(assembledTrainDF)
    val trainAccuracy = evaluator.evaluate(trainPredictionsDF)

    println(s"Training Accuracy: ${trainAccuracy}")

    // test
    val testIndexedDF = indexerModel.transform(testCleanDF).na.fill(0.0)
    val assembledTestDF = assembler.transform(testIndexedDF)
    val testPredictionsDF = model.transform(assembledTestDF)

    testIndexedDF.describe().show()
//    assembledTestDF.describe().show()
//    testPredictionsDF.describe().show()

    testPredictionsDF
      .select("PassengerId", "prediction")
      .withColumnRenamed("prediction", "Survived")
      .coalesce(1)
      .write
      .option("header", "true")
      .csv("src/main/resources/predictions")
  }
}