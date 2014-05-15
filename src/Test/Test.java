package Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class Test {

	public Test() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// train and test on mutation_train.arff
		boolean task1 = false;
		boolean task2 = true;
		boolean task3 = false;

		// 10 cross fold validation on mutation_train.arff
		if (task1) {
			Classifier m_classifier = (Classifier) new NaiveBayes();

			File inputFile = new File(
					"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet();
			inputFile = new File(
					"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // 读入测试文件
			int clsNum = instancesTest.numAttributes();
			// instancesTest.
			instancesTest.setClassIndex(instancesTest.numAttributes() - 1); // 设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数

			double sum = instancesTest.numInstances(), // 测试语料实例数
			right = 0.0f;
			instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);

			m_classifier.buildClassifier(instancesTrain); // 训练

			for (int i = 0; i < sum; i++)// 测试分类结果
			{
				Instance instance = instancesTest.instance(i);
				double[] conf = m_classifier.distributionForInstance(instance);
				double pre = m_classifier.classifyInstance(instance);
				double tru = instancesTest.instance(i).classValue();

				if (Math.abs(pre - tru) < 0.001)// 如果预测值和答案值相等（测试语料中的分类列提供的须为正确答案，结果才有意义）
				{
					right++;// 正确值加1
				}
			}
			System.out.println("J48 classification precision:" + (right / sum));

		}

		if (task2) {
			Classifier myClassifier = (Classifier) new NaiveBayes();
			// File inputFile = new
			// File("C://Program Files//Weka-3-6//data//iris.arff");//训练语料文件
			File inputFile = new File(
					"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances data = atf.getDataSet(); // 读入训练文件
			data.setClassIndex(data.numAttributes() - 1);

			int seed = 3984; // the seed for randomizing the data
			int folds = 10; // the number of folds to generate, >=2

			Random rand = new Random(seed); // create seeded number generator
			Instances randData = new Instances(data); // create copy of original
														// data
			randData.randomize(rand); // randomize data with number generator

			Evaluation eval = new Evaluation(randData);
			eval.crossValidateModel(myClassifier, randData, folds, new Random(
					seed));
			System.out.println(eval.toSummaryString("\nResults\n======\n",
					false));
			System.out.println(eval.toMatrixString());
		}

		if (task3) {
			boolean isBootstrapping = true;
			// other options
			int seed = 7498; // the seed for randomizing the data
			int folds = 5; // the number of folds to generate, >=2
			double threshold = 0.95;

			File inputFile = new File(
					"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances data2 = atf.getDataSet(); // 读入训练文件
			Instances data = new Instances(data2, 0, 100);
			data.setClassIndex(data.numAttributes() - 1);

			inputFile = new File(
					"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_unknown.arff");// 测试语料文件
			atf.setFile(inputFile);
			Instances instancesUnknown = atf.getDataSet(); // read in
															// unannotated
															// instances
			instancesUnknown
					.setClassIndex(instancesUnknown.numAttributes() - 1);

			// classifier
			Classifier cls = (Classifier) new NaiveBayes();
			// Classifier cls = (Classifier) new Logistic();

			// randomize data
			Random rand = new Random(seed);
			Instances randData = new Instances(data);
			randData.randomize(rand);

			// perform cross-validation
			Evaluation eval = new Evaluation(randData);
			double[] pres = new double[folds];


				double preSum = 0.0;
				for (int n = 0; n < folds; n++) {
					Instances instancesTrain = randData.trainCV(folds, n);
					Instances instancesTest = randData.testCV(folds, n);

					// build and evaluate classifier
					Classifier clsCopy = Classifier.makeCopy(cls);
					clsCopy.buildClassifier(instancesTrain);
					eval.evaluateModel(clsCopy, instancesTest);

//					if (isBootstrapping) {
//						clsCopy = bootstrapping(clsCopy, instancesTrain,
//						// new Instances(instancesUnknown, 0, 500), threshold);
//								instancesUnknown, threshold);
//					}

					int sum = instancesTest.numInstances();// 测试语料实例数
					double right = 0.0f;
					for (int i = 0; i < sum; i++)// 测试分类结果
					{
						Instance instance = instancesTest.instance(i);
						double[] conf = clsCopy
								.distributionForInstance(instance);
						double pre = clsCopy.classifyInstance(instance);
						double tru = instancesTest.instance(i).classValue();

						if (Math.abs(pre - tru) < 0.001)// 如果预测值和答案值相等（测试语料中的分类列提供的须为正确答案，结果才有意义）
						{
							right++;// 正确值加1
						}
					}
					double pre = (right / sum);
					preSum += pre;

					pres[n] = pre;
					System.out.println("fold " + n
							+ " classification precision:" + pre);
				}
				System.out.println(folds
						+ " folds classification average precision:"
						+ (preSum / folds));
				System.out.println();

				 //output evaluation
				 System.out.println();
				 System.out.println("=== Setup ===");
				 System.out.println("Classifier: " + cls.getClass().getName()
				 + " "
				 + Utils.joinOptions(cls.getOptions()));
				 System.out.println("Dataset: " + data.relationName());
				 System.out.println("Folds: " + folds);
				 System.out.println("Seed: " + seed);
				 System.out.println();
				 System.out.println(eval.toSummaryString("=== " + folds
				 + "-fold Cross-validation ===", true));
				isBootstrapping = true;
			}


	}

	private static Classifier bootstrapping(Classifier clsCopy,
			Instances instancesTrain, Instances instancesUnknown,
			double threshold) throws Exception {
		boolean isChanged = true;
		int count = 0;
		while (isChanged) {
			isChanged = false;
			int i = 0;
			int sum = instancesUnknown.numInstances();
			while (i < sum) {
				// System.out.println(i);
				Instance unknownIns = instancesUnknown.instance(i);
				double[] conf = clsCopy.distributionForInstance(unknownIns);
				double pre = clsCopy.classifyInstance(unknownIns);

				if (conf[0] > threshold && conf[1] < threshold) {
					// System.out.println("case 1");
					// System.out.println("count" + count);
					if (count == 1489) {
						// System.out.println();
					}
					unknownIns.setClassValue(0.0);
					instancesTrain.add(unknownIns);
					instancesUnknown.delete(i);
					sum--;
					i--;
					isChanged = true;

				}

				if (conf[0] < threshold && conf[1] > threshold) {
					// System.out.println("case 2");
					// System.out.println("count" + count);
					unknownIns.setClassValue(1.0);
					instancesTrain.add(unknownIns);
					// System.out.println(i);
					instancesUnknown.delete(i);
					sum--;
					i--;
					isChanged = true;

				}

				i++;
				count++;
			}
			// clsCopy = (Classifier) new NaiveBayes();
			// clsCopy = (Classifier) new Logistic();
			clsCopy.buildClassifier(instancesTrain);
		}

		return clsCopy;
	}

}
