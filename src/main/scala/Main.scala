import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {
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
    //    trainDF.groupBy("Survived").count().show()
    val trainWithTitleDF = trainDF.withColumn("Title", regexp_extract(col("Name"), "(?<=, ).*?(?=[.])", 0))
    val testWithTitleDF = testDF.withColumn("Title", regexp_extract(col("Name"), "(?<=, ).*?(?=[.])", 0))

    val trainWithFamilySizeDF = trainWithTitleDF.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    val testWithFamilySizeDF = testWithTitleDF.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)

    val trainWithIsAloneDF = trainWithFamilySizeDF.withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))
    val testWithIsAloneDF = testWithFamilySizeDF.withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))

    val trainCleanDF = trainWithIsAloneDF.drop("PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch")
    val testCleanDF = testWithIsAloneDF.drop("PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch")



    trainCleanDF.show()

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

    //???
    val indexerModel = stringIndexer.fit(trainCleanDF)
    val trainIndexedDF = indexerModel.transform(trainCleanDF)

    val assembledTrainDF = assembler.transform(trainIndexedDF)

    val model = lr.fit(assembledTrainDF)

    val trainPredictionsDF = model.transform(assembledTrainDF)

    val trainAccuracy = evaluator.evaluate(trainPredictionsDF)

    println(s"Training Accuracy: ${trainAccuracy}")
  }
}