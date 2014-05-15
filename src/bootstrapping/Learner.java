package bootstrapping;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Learner {

	public static void balancedSetExp() throws Exception {
		// int seed = rand(); // the seed for randomizing the data
		int seed = (int) System.currentTimeMillis();
		int[] seeds = { 1, 384, 748, 28, 84, 263, 29, 264, 45789, 67 };
		int folds = 10; // the number of folds to generate, >=2
		int numExp = 10;
		// boolean isFlip = false;

		File inputFile;
		ArffLoader atf = new ArffLoader();

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train_class1.arff");
		atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instanceClass1 = atf.getDataSet();
		instanceClass1.setClassIndex(instanceClass1.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train_class2.arff");
		atf.setFile(inputFile);
		Instances instanceClass2 = atf.getDataSet();
		instanceClass2.setClassIndex(instanceClass2.numAttributes() - 1);

		Instances instanceKnown = makeBalancedSet(instanceClass1,
				instanceClass2, 25);

		for (int e = 0; e < numExp; e++) {
			seed = seeds[e];
			Random rand = new Random(seed);
			Instances randData = new Instances(instanceKnown);
			randData.randomize(rand);

			NaiveBayes classifier = new NaiveBayes();
			double correct = 0;
			double incorrect = 0;
			// Evaluation eval = new Evaluation(randData);
			for (int run = 0; run < 2; run++) {
				for (int n = 0; n < folds; n++) {
					Instances train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);
					if (run == 1) {
						test = flip(test);
					}
					Classifier clsCopy = Classifier.makeCopy(classifier);
					clsCopy.buildClassifier(train);
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(clsCopy, test);
					// System.out.println(test.numInstances());
					correct += eval.correct();
					incorrect += eval.incorrect();
					// System.out.println(n);
					// System.out.println(correct);
					// System.out.println(incorrect);
				}
				double precision = correct / (correct + incorrect);
				System.out.println(String.format("%.3f", precision));
			}
			System.out.println(e);
		}
		// eval.evaluateModel(classifier, randData);
		// System.out.println(eval.toClassDetailsString());
		// System.out.println(eval.toSummaryString());
		// System.out.println(eval.toMatrixString());

		// Evaluation eval = new Evaluation(randData);
		// eval.crossValidateModel( classifier, randData, 10, new Random(1));
		// System.out.println(eval.toClassDetailsString());
		// System.out.println(eval.toSummaryString());
		// System.out.println(eval.toMatrixString());
	}

	// Number of Seeds vs. Accuracy
	public static void expendingLabeledSetExp(boolean isBootstrapping, boolean isBalanced, int balancedSetSize,
			int labeledSetSize, int[] numSeed) throws Exception {
		// Parameters
		int seed = 7498; // the seed for randomizing the data
		int folds = 10; // the number of folds to generate, >=2
		double threshold = 0.95;

		File inputFile;
		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instanceKnownAll = atf.getDataSet(); // training

		Random rand = new Random(seed);
		Instances randData = new Instances(instanceKnownAll);
		randData.randomize(rand);

		Instances instanceKnown = new Instances(randData, 0, labeledSetSize);
		instanceKnown.setClassIndex(instanceKnown.numAttributes() - 1);
		
		
		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train_class1.arff");
		atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instanceClass1 = atf.getDataSet();
		instanceClass1.setClassIndex(instanceClass1.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train_class2.arff");
		atf.setFile(inputFile);
		Instances instanceClass2 = atf.getDataSet();
		instanceClass2.setClassIndex(instanceClass2.numAttributes() - 1);

		if (isBalanced) {
			instanceKnown = makeBalancedSet(instanceClass1, instanceClass2,
					balancedSetSize);
			System.out.println(String.format("Balanced set (size: %d)",
					instanceKnown.numInstances()));
		} else {
			System.out.println("Unbalanced set");
		}
		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_unknown.arff");
		atf.setFile(inputFile);
		Instances instancesUnknown = atf.getDataSet();
		instancesUnknown.setClassIndex(instancesUnknown.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\neighbors1_train.arff");
		atf.setFile(inputFile);
		Instances instancesNeighbors1 = atf.getDataSet();
		instancesNeighbors1.setClassIndex(instancesUnknown.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\neighbors2_train.arff");
		atf.setFile(inputFile);
		Instances instancesNeighbors2 = atf.getDataSet();
		instancesNeighbors2.setClassIndex(instancesUnknown.numAttributes() - 1);

		Instances instancesNeighbors = new Instances(instancesNeighbors1);
		for (int i = 0; i < instancesNeighbors2.numInstances(); i++) {
			instancesNeighbors.add(instancesNeighbors2.instance(i));
		}

		int[] sizes = new int[] { 5, 10, 16, 20, 25, 40, 50, 80, 100, 200 };
		sizes = new int[] { 2, 3, 4, 5, 10, 25, 40, 50, 80, 100, 200 };
		// sizes = new int[] { 2,3,4,5, 10, 25, 40, 50};
		sizes = new int[] { 2, 4, 5, 10, 25 };
		// int[] sizes = new int[] {10, 20, 40, 50, 80, 100, 200};
		// int[] sizes = new int[] {5, 10, 16};
		// int[] sizes = new int[] {10, 20, 40, 50, 80};
		// int[] sizes = new int[] {100, 200};
		sizes = numSeed;

		Bootstrapping learner = new Bootstrapping();

		List<List<List<Double>>> allAveraged = new ArrayList<List<List<Double>>>();
		for (int i = 0; i < sizes.length; i++) {
			System.out.println(i);
			int size = sizes[i];
			int n = instanceKnown.numInstances() / size;

			List<List<List<Double>>> allResults = new ArrayList<List<List<Double>>>();

			for (int i1 = 0; i1 < n; i1++) {
				Instances seeds = new Instances(instanceKnown.testCV(n, i1));
				Instances testSet = new Instances(instanceKnown.trainCV(n, i1));

				Instances unknownSet = new Instances(instancesUnknown);
				List<List<Double>> res = learner.run(isBootstrapping,
						threshold, seeds, instancesNeighbors, testSet,
						unknownSet);
				allResults.add(res);
			}

			List<List<Double>> averaged = computeAverage(allResults);
			allAveraged.add(averaged);
		}

		// get x axis
		List<Number> xData = new ArrayList<Number>();

		// Precision
		List<Number> run1PreCls1Data = new ArrayList<Number>();
		List<Number> run1PreCls2Data = new ArrayList<Number>();

		List<Number> run2PreCls1Data = new ArrayList<Number>();
		List<Number> run2PreCls2Data = new ArrayList<Number>();

		List<Number> run3PreCls1Data = new ArrayList<Number>();
		List<Number> run3PreCls2Data = new ArrayList<Number>();

		// Recall
		List<Number> run1RecCls1Data = new ArrayList<Number>();
		List<Number> run1Reccls2Data = new ArrayList<Number>();

		List<Number> run2RecCls1Data = new ArrayList<Number>();
		List<Number> run2RecCls2Data = new ArrayList<Number>();

		List<Number> run3RecCls1Data = new ArrayList<Number>();
		List<Number> run3RecCls2Data = new ArrayList<Number>();

		// Accuracy
		List<Number> acy1Data = new ArrayList<Number>();
		List<Number> acy2Data = new ArrayList<Number>();
		List<Number> acy3Data = new ArrayList<Number>();

		for (int i = 0; i < allAveraged.size(); i++) {
			List<List<Double>> avgs = allAveraged.get(i);
			List<Double> res1 = avgs.get(0);
			List<Double> res2 = avgs.get(1);
			List<Double> res3 = avgs.get(2);

			xData.add(res1.get(5));

			run1PreCls1Data.add(res1.get(0));
			run1PreCls2Data.add(res1.get(1));

			run1RecCls1Data.add(res1.get(2));
			run1Reccls2Data.add(res1.get(3));

			run2PreCls1Data.add(res2.get(0));
			run2PreCls2Data.add(res2.get(1));

			run2RecCls1Data.add(res2.get(2));
			run2RecCls2Data.add(res2.get(3));

			run3PreCls1Data.add(res3.get(0));
			run3PreCls2Data.add(res3.get(1));

			run3RecCls1Data.add(res3.get(2));
			run3RecCls2Data.add(res3.get(3));

			acy1Data.add(res1.get(4));
			acy2Data.add(res2.get(4));
			acy3Data.add(res3.get(4));
		}

		printList(xData, "numSeed.txt");

		printList(run1PreCls1Data, "run1PreCls1Data.txt");
		printList(run1PreCls2Data, "run1PreCls2Data.txt");
		printList(run1RecCls1Data, "run1RecCls1Data.txt");
		printList(run1Reccls2Data, "run1Reccls2Data.txt");

		printList(run2PreCls1Data, "run2PreCls1Data.txt");
		printList(run2PreCls2Data, "run2PreCls2Data.txt");
		printList(run2RecCls1Data, "run2RecCls1Data.txt");
		printList(run2RecCls2Data, "run2RecCls2Data.txt");

		printList(run3PreCls1Data, "run3PreCls1Data.txt");
		printList(run3PreCls2Data, "run3PreCls2Data.txt");
		printList(run3RecCls1Data, "run3RecCls1Data.txt");
		printList(run3RecCls2Data, "run3RecCls2Data.txt");

		printList(acy1Data, "acy1Data.txt");
		printList(acy2Data, "acy2Data.txt");
		printList(acy3Data, "acy3Data.txt");

		Plotter myPlt = new Plotter();

		List<List<Number>> yDataList = new ArrayList<List<Number>>();
		yDataList.add(acy1Data);
		yDataList.add(acy2Data);
		yDataList.add(acy3Data);

		List<String> seriesNames = new ArrayList<String>();
		seriesNames.add("without bootstrapping");
		seriesNames.add("with bootstrapping");
		if (isBootstrapping) {
			seriesNames.add("with expand labeled set (with bootstrapping)");
		} else {
			seriesNames.add("with expand labeled set (without bootstrapping)");
		}

		String fileName = "";
		if (isBootstrapping) {
			fileName = String.format(
					"expendingLabeledSetExp_%.2f_with bootstrapping.png",
					threshold);
		} else {
			fileName = String.format(
					"expendingLabeledSetExp_%.2f_without bootstrapping.png",
					threshold);
		}
		myPlt.makePlot(fileName, "Number of Seeds vs. Accuracy",
				"Number of seeds", "Accuracy", xData, yDataList, seriesNames,
				false, true);

		// myPlt.makePlot2("precision vs recall.png", cls1Rec1Data,
		// cls1Pre1Data, cls1Rec2Data, cls1Pre2Data,
		// "Averaged precision vs Averaged recall of class 1",
		// "Averaged Recall", "Averaged Precision", "classifier 1",
		// "classifier 2", false, true);
		// myPlt.makePlot2("precision vs recall.png", cls2Rec1Data,
		// cls2Pre1Data, cls2Rec2Data, cls2Pre2Data,
		// "Averaged precision vs Averaged recall of class 2",
		// "Averaged Recall", "Averaged Precision", "classifier 1",
		// "classifier 2", false, true);

		// public void makePlot(String name, List<Number> xData1,
		// List<Number> yData1, List<Number> yData2, String title, String
		// xTitle, String yTitle,
		// String seriesName, boolean isFixedRadio, boolean isSave) {

		System.out.println();
	}

	// Threshold vs. Accuracy
	public static void exp2() throws Exception {
		boolean isBootstrapping = true;
		int seed = 7498; // the seed for randomizing the data
		int folds = 10; // the number of folds to generate, >=2
		double threshold = 0.95;

		File inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instanceKnownAll = atf.getDataSet(); // training

		Random rand = new Random(seed);
		Instances randData = new Instances(instanceKnownAll);
		randData.randomize(rand);

		Instances instanceKnown = new Instances(randData, 0, 400);
		instanceKnown.setClassIndex(instanceKnown.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_unknown.arff");
		atf.setFile(inputFile);
		Instances instancesUnknown = atf.getDataSet();
		instancesUnknown.setClassIndex(instancesUnknown.numAttributes() - 1);

		double[] thresholds = new double[] { 0.5, 0.55, 0.60, 0.65, 0.7, 0.75,
				0.80, 0.85, 0.9, 0.95, 0.97, 0.99, 0.999 };

		Bootstrapping learner = new Bootstrapping();

		List<List<List<Double>>> allAveraged = new ArrayList<List<List<Double>>>();
		for (int i = 0; i < thresholds.length; i++) {
			threshold = thresholds[i];
			int size = 40;
			int n = instanceKnown.numInstances() / size;

			List<List<List<Double>>> allResults = new ArrayList<List<List<Double>>>();

			for (int i1 = 0; i1 < n; i1++) {
				Instances seeds = new Instances(instanceKnown.testCV(n, i1));
				Instances testSet = new Instances(instanceKnown.trainCV(n, i1));
				Instances unknownSet = new Instances(instancesUnknown);
				List<List<Double>> res = learner.run(isBootstrapping,
						threshold, seeds, null, testSet, unknownSet);
				allResults.add(res);
			}

			List<List<Double>> averaged = computeAverage(allResults);
			allAveraged.add(averaged);
		}

		// get x axis
		List<Number> xData = new ArrayList<Number>();

		List<Number> cls1Pre1Data = new ArrayList<Number>();
		List<Number> cls2Pre1Data = new ArrayList<Number>();
		List<Number> cls1Pre2Data = new ArrayList<Number>();
		List<Number> cls2Pre2Data = new ArrayList<Number>();

		List<Number> cls1Rec1Data = new ArrayList<Number>();
		List<Number> cls2Rec1Data = new ArrayList<Number>();
		List<Number> cls1Rec2Data = new ArrayList<Number>();
		List<Number> cls2Rec2Data = new ArrayList<Number>();

		List<Number> acy1Data = new ArrayList<Number>();
		List<Number> acy2Data = new ArrayList<Number>();

		for (int i = 0; i < allAveraged.size(); i++) {
			List<List<Double>> avgs = allAveraged.get(i);
			List<Double> res1 = avgs.get(0);
			List<Double> res2 = avgs.get(1);
			xData.add(thresholds[i]);

			cls1Pre1Data.add(res1.get(0));
			cls2Pre1Data.add(res1.get(1));

			cls1Pre2Data.add(res2.get(0));
			cls2Pre2Data.add(res2.get(1));

			cls1Rec1Data.add(res1.get(2));
			cls2Rec1Data.add(res1.get(3));

			cls1Rec2Data.add(res2.get(2));
			cls2Rec2Data.add(res2.get(3));

			acy1Data.add(res1.get(4));
			acy2Data.add(res2.get(4));
		}

		printList(xData, "exp2_0.txt");

		printList(cls1Pre1Data, "exp2_1.txt");
		printList(cls2Pre1Data, "exp2_2.txt");
		printList(cls1Pre2Data, "exp2_3.txt");
		printList(cls2Pre2Data, "exp2_4.txt");

		printList(cls1Rec1Data, "exp2_5.txt");
		printList(cls2Rec1Data, "exp2_6.txt");
		printList(cls1Rec2Data, "exp2_7.txt");
		printList(cls2Rec2Data, "exp2_8.txt");

		printList(acy1Data, "exp2_9.txt");
		printList(acy2Data, "exp2_10.txt");

		Plotter myPlt = new Plotter();
		myPlt.makePlot("ThresholdVSAccuracy.png", xData, acy1Data, acy2Data,
				"Threshold vs. Accuracy", "Threshold", "Accuracy",
				"without bootstrapping", "with bootstrapping", false, true);

		System.out.println();
	}

	public static void exp3(int size) throws Exception {
		boolean isBootstrapping = true;
		int seed = 7498; // the seed for randomizing the data
		double threshold = 0.95;

		File inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instanceKnownAll = atf.getDataSet(); // training

		Random rand = new Random(seed);
		Instances randData = new Instances(instanceKnownAll);
		randData.randomize(rand);

		Instances instanceKnown = new Instances(randData, 0, 400);
		instanceKnown.setClassIndex(instanceKnown.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_unknown.arff");
		atf.setFile(inputFile);
		Instances instancesUnknown = atf.getDataSet();
		instancesUnknown.setClassIndex(instancesUnknown.numAttributes() - 1);

		// double[] thresholds = new double[] {0.5, 0.55, 0.60, 0.65, 0.7, 0.75,
		// 0.80, 0.85, 0.9, 0.95, 0.97, 0.99, 0.999};
		int[] numUnknown = new int[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
				150, 200, 300, 500 };

		Bootstrapping learner = new Bootstrapping();

		List<List<List<Double>>> allAveraged = new ArrayList<List<List<Double>>>();
		for (int i = 0; i < numUnknown.length; i++) {
			// threshold = thresholds[i];
			// int size = 20;
			int n = instanceKnown.numInstances() / size;

			List<List<List<Double>>> allResults = new ArrayList<List<List<Double>>>();

			for (int i1 = 0; i1 < n; i1++) {
				Instances seeds = new Instances(instanceKnown.testCV(n, i1));
				Instances testSet = new Instances(instanceKnown.trainCV(n, i1));
				Instances unknownSet = new Instances(instancesUnknown, 0,
						numUnknown[i]);
				List<List<Double>> res = learner.run(isBootstrapping,
						threshold, seeds, null, testSet, unknownSet);
				allResults.add(res);
			}

			List<List<Double>> averaged = computeAverage(allResults);
			allAveraged.add(averaged);
		}

		// get x axis
		List<Number> xData = new ArrayList<Number>();

		List<Number> cls1Pre1Data = new ArrayList<Number>();
		List<Number> cls2Pre1Data = new ArrayList<Number>();
		List<Number> cls1Pre2Data = new ArrayList<Number>();
		List<Number> cls2Pre2Data = new ArrayList<Number>();

		List<Number> cls1Rec1Data = new ArrayList<Number>();
		List<Number> cls2Rec1Data = new ArrayList<Number>();
		List<Number> cls1Rec2Data = new ArrayList<Number>();
		List<Number> cls2Rec2Data = new ArrayList<Number>();

		List<Number> acy1Data = new ArrayList<Number>();
		List<Number> acy2Data = new ArrayList<Number>();

		for (int i = 0; i < allAveraged.size(); i++) {
			List<List<Double>> avgs = allAveraged.get(i);
			List<Double> res1 = avgs.get(0);
			List<Double> res2 = avgs.get(1);
			List<Double> res3 = avgs.get(2);
			xData.add(numUnknown[i]);

			cls1Pre1Data.add(res1.get(0));
			cls2Pre1Data.add(res1.get(1));

			cls1Pre2Data.add(res2.get(0));
			cls2Pre2Data.add(res2.get(1));

			cls1Rec1Data.add(res1.get(2));
			cls2Rec1Data.add(res1.get(3));

			cls1Rec2Data.add(res2.get(2));
			cls2Rec2Data.add(res2.get(3));

			acy1Data.add(res1.get(4));
			acy2Data.add(res2.get(4));
		}

		printList(xData, "exp3_0.txt");

		printList(cls1Pre1Data, "exp3_1.txt");
		printList(cls2Pre1Data, "exp3_2.txt");
		printList(cls1Pre2Data, "exp3_3.txt");
		printList(cls2Pre2Data, "exp3_4.txt");

		printList(cls1Rec1Data, "exp3_5.txt");
		printList(cls2Rec1Data, "exp3_6.txt");
		printList(cls1Rec2Data, "exp3_7.txt");
		printList(cls2Rec2Data, "exp3_8.txt");

		printList(acy1Data, "exp3_9.txt");
		printList(acy2Data, "exp3_10.txt");

		Plotter myPlt = new Plotter();
		myPlt.makePlot("NumberOfUnlabeledVSAccuracy.png", xData, acy1Data,
				acy2Data, "Number of Unlabeled Instances vs. Accuracy",
				"Number of unlabeled instances", "Accuracy",
				"without bootstrapping", "with bootstrapping", false, true);

		System.out.println();
	}

	public static void exp4() throws Exception {
		File inputFile;
		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\neighbors1_train.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances train1 = atf.getDataSet();
		train1.setClassIndex(train1.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\neighbors2_train.arff");
		atf.setFile(inputFile);
		Instances train2 = atf.getDataSet();
		train2.setClassIndex(train2.numAttributes() - 1);

		Instances train = new Instances(train1);
		for (int i = 0; i < train2.numInstances(); i++) {
			train.add(train2.instance(i));
		}

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train.arff");
		atf.setFile(inputFile);
		Instances test = atf.getDataSet();
		test.setClassIndex(test.numAttributes() - 1);

		Classifier myCls = (Classifier) new NaiveBayes();
		myCls.buildClassifier(train);

		Evaluation eval1 = new Evaluation(train);
		eval1.evaluateModel(myCls, test);

		System.out.println(eval1.toSummaryString(
				"\nClassidier Results\n======\n", false));

		int sum = test.numInstances();
		int error = 0;
		for (int i = 0; i < test.numInstances(); i++)// 测试分类结果
		{
			double predicted = myCls.classifyInstance(test.instance(i));
			double trueLabel = test.instance(i).classValue();
			if (predicted != trueLabel) {
				error++;
			}
			System.out.println(String.format(
					"[%d]True: %f, Predicted: %f, Error: %d", i, trueLabel,
					predicted, error));
		}
		System.out.println("Precision: "
				+ (((double) (sum - error)) / ((double) sum)));

	}

	public static void printList(List<Number> list, String fileName)
			throws IOException {
		StringBuilder sb = new StringBuilder();
		for (Number num : list) {
			sb.append(num.doubleValue());
			sb.append(", ");
		}
		System.out.println(sb.toString());

		String timestamp = Utility.getTimeStamp();
		fileName = timestamp + "_" + fileName;

		File file = new File(fileName);

		// if file doesnt exists, then create it
		if (!file.exists()) {
			file.createNewFile();
		}

		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(sb.toString());
		bw.close();
	}

	public static List<List<Double>> computeAverage(
			List<List<List<Double>>> allResults) {

		List<Double> res1 = new ArrayList<Double>(allResults.get(0).get(0));
		List<Double> res2 = new ArrayList<Double>(allResults.get(0).get(1));
		List<Double> res3 = new ArrayList<Double>(allResults.get(0).get(2));

		for (int i = 1; i < allResults.size(); i++) {
			List<List<Double>> resTemp = allResults.get(i);
			addList(res1, resTemp.get(0));
			addList(res2, resTemp.get(1));
			addList(res3, resTemp.get(2));
		}
		divideList(res1, allResults.size());
		divideList(res2, allResults.size());
		divideList(res3, allResults.size());

		List<List<Double>> res = new ArrayList<List<Double>>();
		res.add(res1);
		res.add(res2);
		res.add(res3);

		return res;
	}

	public static void addList(List<Double> a, List<Double> b) {
		for (int i = 0; i < a.size(); i++) {
			a.set(i, a.get(i) + b.get(i));
		}
	}

	public static void divideList(List<Double> a, double b) {
		for (int i = 0; i < a.size(); i++) {
			a.set(i, a.get(i) / b);
		}
	}

	public static void exp5() throws Exception {
		File inputFile;
		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train_class1.arff");
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instanceClass1 = atf.getDataSet();
		instanceClass1.setClassIndex(instanceClass1.numAttributes() - 1);

		inputFile = new File(
				"C:\\Users\\Dongye\\Dropbox\\Phenoscape\\JCI\\data\\mutation_train_class2.arff");
		atf.setFile(inputFile);
		Instances instanceClass2 = atf.getDataSet();
		instanceClass2.setClassIndex(instanceClass2.numAttributes() - 1);

		Instances randData;

		randData = makeBalancedSet(instanceClass1, instanceClass2, 39);
		Instances test = new Instances(randData.testCV(3, 1));
		Instances train = new Instances(randData.trainCV(3, 1));

		Classifier myCls1 = (Classifier) new NaiveBayes();
		myCls1.buildClassifier(train);
		Evaluation eval1 = new Evaluation(train);
		eval1.evaluateModel(myCls1, test);
		Utility myUtility = new Utility();
		List<Double> res = myUtility.getStatistics(eval1);
		System.out.println(res);
		System.out.println();
	}

	public static Instances makeBalancedSet(Instances instanceClass1,
			Instances instanceClass2, int halfSize) {
		int seed1 = 7249; // the seed for randomizing the data
		int seed2 = 98;

		Instances randData;
		Instances smallSet;
		Instances largeSet;

		int size1 = instanceClass1.numInstances();
		int size2 = instanceClass2.numInstances();
		int smallSize = Math.min(size1, size2);

		if (size1 < size2) {
			smallSet = instanceClass1;
			largeSet = instanceClass2;
		} else {
			smallSet = instanceClass2;
			largeSet = instanceClass1;
		}

		Random rand2 = new Random(seed2);
		largeSet.randomize(rand2);

		randData = new Instances(smallSet);
		for (int i = 0; i < smallSize; i++) {
			randData.add(largeSet.instance(i));
		}

		Random rand1 = new Random(seed1);
		randData.randomize(rand1);

		return randData;
	}

	public static Instances flip(Instances insList) {
		int size = insList.numInstances();

		for (int i = 0; i < size; i++) {
			Instance ins = insList.instance(i);
			double cls = ins.classValue();
			if (isEqualDouble(cls, 0)) {
				ins.setClassValue(1);
			} else {
				ins.setClassValue(0);
			}
		}

		return insList;
	}

	public static boolean isEqualDouble(double d1, double d2) {
		double epsilon = 0.00000001;
		if (Math.abs(d1 - d2) < epsilon) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		boolean exp3 = false;
		boolean exp4 = false;
		boolean exp5 = true;

		boolean isBootstrapping = true;
		int labeledSetSize = 400;
		int[] seeds = new int[] { 5, 10, 16, 20, 25, 40, 50, 80, 100, 200 };
		// seeds = new int[] { 10};
		boolean isBalanced = false;
		int balancedSetSize = 30;

		if (exp3) {
			for (int i = 0; i < 2; i++) {
				isBootstrapping = (i == 1);
				System.out.println(String.format("Run: %d", i));
				expendingLabeledSetExp(isBootstrapping, isBalanced,
						balancedSetSize, labeledSetSize, seeds);
			}
		}

		if (exp4) {
			seeds = new int[] { 2, 3, 5, 10, 15, 20, 30 };
			isBalanced = true;
			for (int i = 0; i < 2; i++) {
				isBootstrapping = (i == 1);
				System.out.println(String.format("Run: %d", i));
				expendingLabeledSetExp(isBootstrapping, isBalanced,
						balancedSetSize, labeledSetSize, seeds);
			}

		}

		if (exp5) {
			balancedSetExp();
		}
		// exp2();
		// exp3(20);
		// List<Number> a = new ArrayList<Number>();
		// a.add(1.0);
		// a.add(2.0);
		// printList(a, "1.txt");
		// exp4();
		// exp5();
	}

}
