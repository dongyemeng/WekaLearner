package bootstrapping;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Evaluation;

public class Utility {

	public Utility() {
		// TODO Auto-generated constructor stub
	}
	
	public static String getTimeStamp(){
		java.util.Date date= new java.util.Date();
		String timestamp =(new Timestamp(date.getTime())).toString();
		timestamp = timestamp.replaceAll(":", "_");
		timestamp = timestamp.replaceAll(" ", "_");
		timestamp = timestamp.replaceAll("\\.", "_");
		timestamp = timestamp.replaceAll("-", "_");
		
		return timestamp;
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

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
