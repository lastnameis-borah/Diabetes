// Databricks notebook source
// MAGIC %md
// MAGIC #Logistic Regression Model
// MAGIC 
// MAGIC ### Predicting Diabetes
// MAGIC 
// MAGIC  The objective of the dataset is to **diagnostically predict whether or not a patient has diabetes**, based on certain diagnostic measurements included in the dataset (collected from Kaggle).

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Loading Source Data
// MAGIC The data for this project is provided as a CSV file containing details of patient. The data includes specific characteristics (or *features*) for each patient, as well as a column(Outcome) indicating Diabetes or not.
// MAGIC 
// MAGIC Load this data into a DataFrame.

// COMMAND ----------

// DBTITLE 1,Code for Loading Data (csv file) to Dataframe 
// MAGIC %scala 
// MAGIC 
// MAGIC // File location and type
// MAGIC val file_location = "/FileStore/tables/Diabetes.csv"
// MAGIC val file_type = "csv"
// MAGIC 
// MAGIC // CSV options
// MAGIC val infer_schema = "true"
// MAGIC val first_row_is_header = "true"
// MAGIC val delimiter = ","
// MAGIC 
// MAGIC // The applied options are for CSV files. For other file types, these will be ignored.
// MAGIC val diabetesDF = spark.read.format(file_type) 
// MAGIC   .option("inferSchema", infer_schema) 
// MAGIC   .option("header", first_row_is_header) 
// MAGIC   .option("sep", delimiter) 
// MAGIC   .load(file_location)
// MAGIC 
// MAGIC diabetesDF.show()

// COMMAND ----------

// DBTITLE 1,Data Details
// MAGIC %md
// MAGIC 
// MAGIC * Pregnancies: Number of times pregnant
// MAGIC 
// MAGIC * Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
// MAGIC 
// MAGIC * BloodPressure: Diastolic blood pressure (mm Hg)
// MAGIC 
// MAGIC * SkinThickness: Triceps skin fold thickness (mm)
// MAGIC 
// MAGIC * Insulin: 2-Hour serum insulin (mu U/ml)
// MAGIC 
// MAGIC * BMI: Body mass index (weight in kg/(height in m)^2)
// MAGIC 
// MAGIC * Diabetes PedigreeFunction: Diabetes pedigree function
// MAGIC 
// MAGIC * Age: Age (years)
// MAGIC 
// MAGIC * Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0

// COMMAND ----------

// DBTITLE 1,Count Records
// MAGIC %scala
// MAGIC 
// MAGIC diabetesDF.count()

// COMMAND ----------

// DBTITLE 1,Finding count, mean, maximum, standard deviation and minimum
// MAGIC %scala
// MAGIC 
// MAGIC diabetesDF.select("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome").describe().show()

// COMMAND ----------

// DBTITLE 1,Printing Schema
// MAGIC %scala
// MAGIC 
// MAGIC diabetesDF.printSchema()

// COMMAND ----------

// DBTITLE 1,Creating Temp View from Dataframe 
// MAGIC %scala
// MAGIC 
// MAGIC diabetesDF.createOrReplaceTempView("DiabetesData");

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select * from DiabetesData; 

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #Exploratory Data Analysis

// COMMAND ----------

// DBTITLE 1,Diabetes Result
// MAGIC %sql
// MAGIC 
// MAGIC SELECT count(Outcome),
// MAGIC CASE
// MAGIC     WHEN Outcome == 1 THEN "Having Diabetes"
// MAGIC     ELSE "Not Having Diabetes"
// MAGIC END AS Outcome
// MAGIC FROM DiabetesData group by Outcome;

// COMMAND ----------

// DBTITLE 1,Generalised Visualisation of all Features
// MAGIC %sql
// MAGIC 
// MAGIC select Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome from DiabetesData;

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #Histograms

// COMMAND ----------

// DBTITLE 1,Histogram of Pregnancies
// MAGIC %sql
// MAGIC 
// MAGIC select Pregnancies from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of Glucose
// MAGIC %sql
// MAGIC 
// MAGIC select Glucose from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of Blood Pressure
// MAGIC %sql
// MAGIC 
// MAGIC select BloodPressure from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of SkinThickness
// MAGIC %sql
// MAGIC 
// MAGIC select SkinThickness from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of Insulin
// MAGIC %sql
// MAGIC 
// MAGIC select Insulin from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of BMI
// MAGIC %sql
// MAGIC 
// MAGIC select BMI from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of DiabetesPedigreeFunction
// MAGIC %sql
// MAGIC 
// MAGIC select DiabetesPedigreeFunction from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of Age
// MAGIC %sql
// MAGIC 
// MAGIC select Age from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Histogram of Outcome
// MAGIC %sql
// MAGIC 
// MAGIC select Outcome from DiabetesData;

// COMMAND ----------

// DBTITLE 1,Pregnancies Result
// MAGIC %sql
// MAGIC 
// MAGIC select count(Pregnancies), Pregnancies from DiabetesData group by Pregnancies order by Pregnancies;

// COMMAND ----------

// DBTITLE 1,Pregnancies VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(Pregnancies), Pregnancies, Outcome from DiabetesData group by Pregnancies,Outcome order by Pregnancies;

// COMMAND ----------

// DBTITLE 1,Glucose VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(Glucose), Glucose, Outcome from DiabetesData group by Glucose,Outcome order by Glucose;

// COMMAND ----------

// DBTITLE 1,BloodPressure VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(BloodPressure), BloodPressure, Outcome from DiabetesData group by BloodPressure,Outcome order by BloodPressure;

// COMMAND ----------

// DBTITLE 1,SkinThickness VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(SkinThickness), SkinThickness, Outcome from DiabetesData group by SkinThickness,Outcome order by SkinThickness;

// COMMAND ----------

// DBTITLE 1,Insulin VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(Insulin), Insulin, Outcome from DiabetesData group by Insulin,Outcome order by Insulin;

// COMMAND ----------

// DBTITLE 1,BMI VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(BMI), BMI, Outcome from DiabetesData group by BMI,Outcome order by BMI;

// COMMAND ----------

// DBTITLE 1,DiabetesPedigreeFunction VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(DiabetesPedigreeFunction), DiabetesPedigreeFunction, Outcome from DiabetesData group by DiabetesPedigreeFunction,Outcome order by DiabetesPedigreeFunction;

// COMMAND ----------

// DBTITLE 1,Age VS Diabetes
// MAGIC %sql
// MAGIC 
// MAGIC select count(Age), Age, Outcome from DiabetesData group by Age,Outcome order by Age;

// COMMAND ----------

// DBTITLE 1,SkinThickness VS Insulin
// MAGIC %sql
// MAGIC 
// MAGIC select SkinThickness, Insulin from DiabetesData;

// COMMAND ----------

// MAGIC %md ##Classification Model
// MAGIC 
// MAGIC I have implemented a classification model **(Logistic Regression)** that uses features of Patient details and will predict if the patient is Diabetic or Not
// MAGIC 
// MAGIC ### Import Spark SQL and Spark ML Libraries

// COMMAND ----------

// DBTITLE 1,Importing Libraries
// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.sql.types._
// MAGIC import org.apache.spark.sql.functions._
// MAGIC 
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md ### Preparing the Training Data
// MAGIC To train the classification model, we need a training data set that includes a vector of numeric features, and a label column. I will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **Outcome** column to **label**.

// COMMAND ----------

// MAGIC %md ###VectorAssembler()
// MAGIC 
// MAGIC VectorAssembler():  is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. 
// MAGIC 
// MAGIC **VectorAssembler** accepts the following input column types: **all numeric types, boolean type, and vector type.** 
// MAGIC 
// MAGIC In each row, the **values of the input columns will be concatenated into a vector** in the specified order.

// COMMAND ----------

// MAGIC %md ### Spliting the Data
// MAGIC I will use 70% of the data for training, and reserve 30% for testing. 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val splits = diabetesDF.randomSplit(Array(0.7, 0.3))
// MAGIC val train = splits(0)
// MAGIC val test = splits(1)
// MAGIC val train_rows = train.count()
// MAGIC val test_rows = test.count()
// MAGIC println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// DBTITLE 1,Vector Assembler Code
// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.feature.VectorAssembler
// MAGIC 
// MAGIC val assembler = new VectorAssembler().setInputCols(Array("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin", "BMI", "DiabetesPedigreeFunction", "Age" )).setOutputCol("features")
// MAGIC 
// MAGIC val training = assembler.transform(train).select($"features", $"Outcome".alias("label"))
// MAGIC 
// MAGIC training.show(false)

// COMMAND ----------

// MAGIC %md ### Train a Classification Model (Logistic Regression)
// MAGIC Next, I need to train a Classification Model using the training data. To do this, I'll create an instance of the Logistic regression algorithm and use its **fit** method to train a model based on the training DataFrame. In this Project, I will use a *Logistic Regression* algorithm.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC 
// MAGIC val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(500).setRegParam(0.3)
// MAGIC val model = lr.fit(training)
// MAGIC println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Preparing the Testing Data
// MAGIC Now that the model is trained, I can test it using the testing data you reserved previously. First, I need to prepare the testing data in the same way as I did the training data by transforming the feature columns into a vector. This time I'll rename the **Outcome** column to **trueLabel**.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val testing = assembler.transform(test).select($"features", $"Outcome".alias("trueLabel"))
// MAGIC testing.show(false)

// COMMAND ----------

// MAGIC %md ### Testing the Model
// MAGIC Now I'm ready to use the **transform** method of the model to generate some predictions. But in this case I'll be using the test data which includes a known true label value, so that I can compare the Diabetes result. 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val prediction = model.transform(testing)
// MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
// MAGIC predicted.show(300)

// COMMAND ----------

// MAGIC %md Looking at the result, the **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data. There is some variance between the predictions and the actual values (the individual differences are referred to as *residuals*).

// COMMAND ----------

// MAGIC %md ### Computing Confusion Matrix Metrics
// MAGIC Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
// MAGIC - True Positives
// MAGIC - True Negatives
// MAGIC - False Positives
// MAGIC - False Negatives
// MAGIC 
// MAGIC From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val tp = predicted.filter("prediction == 1 AND truelabel == 1").count().toFloat
// MAGIC val fp = predicted.filter("prediction == 1 AND truelabel == 0").count().toFloat
// MAGIC val tn = predicted.filter("prediction == 0 AND truelabel == 0").count().toFloat
// MAGIC val fn = predicted.filter("prediction == 0 AND truelabel == 1").count().toFloat
// MAGIC   val metrics = spark.createDataFrame(Seq(
// MAGIC  ("TP", tp),
// MAGIC  ("FP", fp),
// MAGIC  ("TN", tn),
// MAGIC  ("FN", fn),
// MAGIC  ("Precision", tp / (tp + fp)),
// MAGIC  ("Recall", tp / (tp + fn)))).toDF("metric", "value")
// MAGIC metrics.show()

// COMMAND ----------

// MAGIC %md ### Review the Area Under ROC
// MAGIC Another way to assess the performance of a classification model is to measure the area under a ROC(Receiver Operating Characteristic) curve for the model. the spark.ml library includes a **BinaryClassificationEvaluator** class that I'll use to compute this. The ROC curve shows the True Positive and False Positive rates plotted for varying thresholds.

// COMMAND ----------

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
val auc = evaluator.evaluate(prediction)
println("AUC = " + (auc))

// COMMAND ----------

// MAGIC %md ### Training a One-vs-Rest classifier (a.k.a. One-vs-All) Model
// MAGIC OneVsRest is an example of a machine learning reduction for performing multiclass classification given a base classifier that can perform binary classification efficiently. It is also known as “One-vs-All.”

// COMMAND ----------

import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// load data file.
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().setInputCols(Array("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin", "BMI", "DiabetesPedigreeFunction", "Age" )).setOutputCol("features")

val training = assembler.transform(diabetesDF).select($"features", $"Outcome".alias("label"))

// generate the train/test split.
val Array(train, test) = training.randomSplit(Array(0.7, 0.3))

// instantiate the base classifier
val classifier = new LogisticRegression()
  .setMaxIter(10)
  .setTol(1E-6)
  .setFitIntercept(true)

// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(classifier)

// train the multiclass model.
val ovrModel = ovr.fit(train)

// score the model on test data.
val predictions = ovrModel.transform(test)

val predicted = predictions.select("features", "prediction", "label")
predicted.show()


// COMMAND ----------

// DBTITLE 1,Evaluation of the Model (accuracy)
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

// compute the classification error on test data.
val accuracy = evaluator.evaluate(predictions)
