/**
 * @author Nils Persson
 * @date 2019-Apr-20 7:52:10 PM 
 */

/**
 * 
 */
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

public class Main {
	
	public <K, V extends Comparable<? super V>> Map<K, V>
    sortByValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list =
                new LinkedList<Map.Entry<K, V>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }
	
	public double getMinorityValue(Instances instances) throws Exception {
        int numInstances = instances.numInstances();
        if (numInstances < 1) {
            throw new Exception("Empty Instances");
        }
        Map<String, Integer> mapValues = new TreeMap<String, Integer>();
        for (int i = 0; i < numInstances; i++) {
            Instance instance = instances.instance(i);
            String key = String.valueOf(instance.value(instance.classAttribute()));
            if (mapValues.containsKey(key)) {
                mapValues.put(String.valueOf(key), (mapValues.get(key) + 1));
            } else {
                mapValues.put(String.valueOf(key), 1);
            }
        }
        mapValues = this.sortByValue(mapValues);
        for (Map.Entry<String, Integer> entry : mapValues.entrySet()) {
//        	System.out.println(Double.valueOf(entry.getKey()));
            return Double.valueOf(entry.getKey());
        }

        return -1.0;
    }
	
	public void separateDecisionClasses(Instances instances, Instances minorityInstances, Instances majorityInstances) throws Exception {
        double minorityValue = getMinorityValue(instances);
        minorityInstances.delete();
        majorityInstances.delete();
        int numInstances = instances.numInstances();
        for (int i = 0; i < numInstances; i++) {
            Instance instance = instances.instance(i);
            if (instance.value(instance.classAttribute()) == minorityValue) {
                minorityInstances.add(instance);
            } else {
                majorityInstances.add(instance);
            }
        }
    }
	
	public double getSMOTEPercentage(Instances instances){
		Instances minorityInstances = new Instances(instances);
        Instances majorityInstances = new Instances(instances);
        Instances newInstances = new Instances(instances);
        try {
            separateDecisionClasses(newInstances, minorityInstances, majorityInstances);
        } catch (Exception e) {
            e.printStackTrace();
        }
//        System.out.println(minorityInstances.numInstances());
        return ((1.0 * majorityInstances.numInstances() / minorityInstances.numInstances()) - 1) * 100.0;
	}
	
	public double[] getEvalResults(String path) throws Exception{
		double[] results = new double[7];
		// Load data  
		DataSource source = new DataSource(path);
		Instances data = source.getDataSet();

		// Set class to last attribute
		if (data.classIndex() == -1)
		    data.setClassIndex(data.numAttributes() - 1);

		// use StratifiedRemoveFolds to randomly split the data  
		StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

		// set options for creating the subset of data
		String[] options = new String[6];

		options[0] = "-N";                 // indicate we want to set the number of folds                        
		options[1] = Integer.toString(5);  // split the data into five random folds
		options[2] = "-F";                 // indicate we want to select a specific fold
		options[3] = Integer.toString(1);  // select the first fold
		options[4] = "-S";                 // indicate we want to set the random seed
		options[5] = Integer.toString((int )(Math.random() * 50 + 1));  // set the random seed to 1

		filter.setOptions(options);        // set the filter options
		filter.setInputFormat(data);       // prepare the filter for the data format    
		filter.setInvertSelection(false);  // do not invert the selection

		// apply filter for test data here
		Instances test = Filter.useFilter(data, filter);

		//  prepare and apply filter for training data here
		filter.setInvertSelection(true);     // invert the selection to get other data 
		Instances train = Filter.useFilter(data, filter);
		
		// use SMOTE filter on training dataset
		int numAttr = train.numAttributes();
		train.setClassIndex(numAttr - 1);
		test.setClassIndex(numAttr - 1);
		
		SMOTE smote = new SMOTE();
		smote.setInputFormat(train);
		smote.setOptions(weka.core.Utils.splitOptions("-C 0 -K 5 -P " + 
					getSMOTEPercentage(train) + " -S 1"));
		
		/** classifiers setting*/
		J48 j48 = new J48();
		//j48.buildClassifier(train);

		FilteredClassifier fc = new FilteredClassifier();
		fc.setClassifier(j48);
		fc.setFilter(smote);
		fc.buildClassifier(train);
		
		Evaluation eval = new Evaluation(train);
//		long start = System.nanoTime();
		eval.evaluateModel(fc, test);
//		long end = System.nanoTime();
//		System.out.println((path.split("/"))[1] + ", " + ((1.0 * end)-start) + ", " + test.numInstances() + ", " + (((1.0 * end)-start)/test.numInstances()));
//		eval.crossValidateModel(j48, data, 10, new Random(1));
		
		// fill in the results
		results[0] = eval.precision(0);
		results[1] = eval.recall(0);
		results[2] = eval.fMeasure(0);
		results[3] = eval.precision(1);
		results[4] = eval.recall(1);
		results[5] = eval.fMeasure(1);
		results[6] = 1-eval.errorRate();
		
		return results;
	}
	
	public double[] getAverageEvalResults(int runs, String path) throws Exception{
		double[] results = new double[7];
		for(int i = 0; i < runs; i++){
			double[] temp = getEvalResults(path);
			for(int j = 0; j < temp.length; j++){
				results[j] = results[j] + temp[j];
			}
		}
		for(int i = 0; i < results.length; i++){
			results[i] = results[i] / runs;
		}
		return results;
	}
	
	public static final int RUNS = 10;
	
	public static final String[] FILES={
			"final_datasets/static_old_10.arff",
			"final_datasets/static_old_20.arff",
			"final_datasets/static_old_40.arff",
			"final_datasets/static_old_60.arff",
			"final_datasets/static_old_80.arff",
			"final_datasets/static_old_100.arff",
			"final_datasets/window_10.arff",
			"final_datasets/window_20.arff",
			"final_datasets/window_40.arff",
			"final_datasets/window_60.arff",
			"final_datasets/window_80.arff",
			"final_datasets/window_100.arff"
	};
	
	public static void main(String[] args) throws Exception{
		Main m = new Main();
		double[] results = new double[7];
		for(int i = 0; i < FILES.length; i++){
			results = m.getAverageEvalResults(RUNS, FILES[i]);
			System.out.print(FILES[i] + ",");
			for(double result : results){
				System.out.print(result + ",");
			}
			System.out.println();
		}
	}
}
