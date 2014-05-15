package bootstrapping;

import java.io.IOException;
import java.util.List;

import com.xeiam.xchart.BitmapEncoder;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.SwingWrapper;
import com.xeiam.xchart.StyleManager.LegendPosition;

public class Plotter {

	public void makePlot(String name, List<Number> xData1, List<Number> yData1,
			List<Number> yData2, String title, String xTitle, String yTitle,
			String seriesName1, String seriesName2, boolean isFixedRadio,
			boolean isSave) {
		// Create Chart
		Chart chart = new ChartBuilder().title(title).xAxisTitle(xTitle)
				.yAxisTitle(yTitle).build();

		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(true);

		// Series
		chart.addSeries(seriesName1, xData1, yData1);
		chart.addSeries(seriesName2, xData1, yData2);
		chart.getStyleManager().setLegendPosition(LegendPosition.OutsideE);

		new SwingWrapper(chart).displayChart();

		if (isSave) {
			try {
				name = Utility.getTimeStamp() + "_" + name;
				BitmapEncoder.savePNG(chart, name);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	public void makePlot(String name, String title, String xTitle,
			String yTitle, List<Number> xData, List<List<Number>> yDataList,
			List<String> seriesNames, boolean isFixedRadio, boolean isSave) {	
		// Create Chart
		Chart chart = new ChartBuilder().title(title).xAxisTitle(xTitle)
				.yAxisTitle(yTitle).build();

		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(true);

		// Series
		int size = yDataList.size(); 
		for (int i = 0; i < size; i++) {
			chart.addSeries(seriesNames.get(i), xData, yDataList.get(i));
		}
		
		chart.getStyleManager().setLegendPosition(LegendPosition.OutsideE);

		new SwingWrapper(chart).displayChart();

		if (isSave) {
			try {
				name = Utility.getTimeStamp() + "_" + name;
				BitmapEncoder.savePNG(chart, name);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}
	
	public void makePlot3(String name, List<Number> xData1, List<Number> yData1,
			List<Number> yData2, List<Number> yData3, String title, String xTitle, String yTitle,
			String seriesName1, String seriesName2, String seriesName3, boolean isFixedRadio,
			boolean isSave) {
		// Create Chart
		Chart chart = new ChartBuilder().title(title).xAxisTitle(xTitle)
				.yAxisTitle(yTitle).build();

		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(true);

		// Series
		chart.addSeries(seriesName1, xData1, yData1);
		chart.addSeries(seriesName2, xData1, yData2);
		chart.addSeries(seriesName3, xData1, yData3);
		chart.getStyleManager().setLegendPosition(LegendPosition.OutsideE);

		new SwingWrapper(chart).displayChart();

		if (isSave) {
			try {
				name = Utility.getTimeStamp() + "_" + name;
				BitmapEncoder.savePNG(chart, name);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public void makePlot2(String name, List<Number> xData1,
			List<Number> xData2, List<Number> yData1, List<Number> yData2,
			String title, String xTitle, String yTitle, String seriesName1,
			String seriesName2, boolean isFixedRadio, boolean isSave) {
		// Create Chart
		Chart chart = new ChartBuilder().title(title).xAxisTitle(xTitle)
				.yAxisTitle(yTitle).build();

		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(true);

		// Series
		chart.addSeries(seriesName1, xData2, yData1);
		chart.addSeries(seriesName2, xData1, yData2);
		chart.getStyleManager().setLegendPosition(LegendPosition.OutsideE);

		new SwingWrapper(chart).displayChart();

		if (isSave) {
			try {
				name = Utility.getTimeStamp() + "_" + name;
				BitmapEncoder.savePNG(chart, name);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public Plotter() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
