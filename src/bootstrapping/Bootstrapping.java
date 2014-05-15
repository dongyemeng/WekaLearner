package bootstrapping;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Bootstrapping {

	public List<List<Double>> run(boolean isBootstrapping, double threshold,
			Instances seeds, Instances neighbors, Instances testSet, Instances unknownSet)
			throws Exception {
		List<List<Double>> res = new ArrayList<List<Double>>();

		// classifier
		Classifier myCls1 = (Classifier) new NaiveBayes();
		Classifier myCls2 = Classifier.makeCopy(myCls1);
		Classifier myCls3 = Classifier.makeCopy(myCls1);

		myCls1.buildClassifier(seeds);

		Evaluation eval1 = new Evaluation(seeds);
		eval1.evaluateModel(myCls1, testSet);
		List<Double> res1 = this.getStatistics(eval1);
		res1.add((double)seeds.numInstances());
//		System.out.println(eval1.toSummaryString("\nClassidier 1 Results\n======\n", false));

		
		res.add(res1);
		
		
//		if (!isBootstrapping) {
//			List<Double> res2 = new ArrayList<Double>();
//			
//			res.add(res2);
//
//			return res;
//		}

		Instances newTrain = bootstrapping(myCls2, seeds, unknownSet, threshold);


		myCls2.buildClassifier(newTrain);
		Evaluation eval2 = new Evaluation(newTrain);
		eval2.evaluateModel(myCls2, testSet);
		
		List<Double> res2 = this.getStatistics(eval2);
		res2.add((double)seeds.numInstances());
//		System.out.println(eval2.toSummaryString("\nClassifier 2 Results\n======\n", false));
		res.add(res2);
		
		
		// Borrow
		Instances expandTrain = new Instances(seeds);
		for (int i = 0; i < neighbors.numInstances(); i++) {
			expandTrain.add(neighbors.instance(i));
		}

		Instances newExpandTrain;
		if (isBootstrapping) {
			newExpandTrain = bootstrapping(myCls3, expandTrain, unknownSet,
					threshold);
		} else {
			newExpandTrain = expandTrain;
		}
		
		myCls3.buildClassifier(newExpandTrain);
		Evaluation eval3 = new Evaluation(newTrain);
		eval3.evaluateModel(myCls3, testSet);
		
		List<Double> res3 = this.getStatistics(eval3);
		res3.add((double) seeds.numInstances());
//		System.out.println(eval2.toSummaryString("\nClassifier 2 Results\n======\n", false));
		res.add(res3);
		
		// 0: pr of cls1
		// 1: pr of cls2
		// 2: recall of cls1
		// 3: recall of cls2
		// 4: accuracy
		// 5: num of seeds
		return res;
	}

	List<Double> getStatistics(Evaluation eval) {
		double p0 = eval.precision(0);
		double p1 = eval.precision(1);
		double r0 = eval.recall(0);
		double r1 = eval.recall(1);
		double acy = eval.correct() / (eval.correct() + eval.incorrect());

		ArrayList<Double> res = new ArrayList<Double>();
		res.add(p0);
		res.add(p1);
		res.add(r0);
		res.add(r1);
		res.add(acy);

		return res;
	}
	


	private Instances bootstrapping(Classifier clsCopy,
			Instances seeds, Instances instancesUnknown,
			double threshold) throws Exception {
		boolean isChanged = true;
		
		Instances instancesTrain = new Instances(seeds);
		clsCopy.buildClassifier(instancesTrain);

		while (isChanged) {
			isChanged = false;
			int i = 0;
			int sum = instancesUnknown.numInstances();
			while (i < sum) {

				Instance unknownIns = instancesUnknown.instance(i);
				double[] conf = clsCopy.distributionForInstance(unknownIns);

				if (conf[0] > threshold && conf[1] < threshold) {
					unknownIns.setClassValue(0.0);
					instancesTrain.add(unknownIns);
					instancesUnknown.delete(i);
					sum--;
					i--;
					isChanged = true;

				}

				if (conf[0] < threshold && conf[1] > threshold) {
					unknownIns.setClassValue(1.0);
					instancesTrain.add(unknownIns);
					instancesUnknown.delete(i);
					sum--;
					i--;
					isChanged = true;

				}

				i++;
			}

			// retrain the classifier
			clsCopy.buildClassifier(instancesTrain);
		}

		return instancesTrain;
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

	}

}
