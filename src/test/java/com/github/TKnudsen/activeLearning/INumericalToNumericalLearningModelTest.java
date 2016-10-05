package com.github.TKnudsen.activeLearning;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.INumericalDataRegression;

/**
 * <p>
 * Title: INumericalToNumericalLearningModelTest
 * </p>
 * 
 * <p>
 * Description: dummy test for the MyNum2NumLearningModel class.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.0
 */
public class INumericalToNumericalLearningModelTest {

	static class MyNum2NumLearningModel implements INumericalDataRegression {
		Double label;

		public void train(NumericalFeatureVector featureVector, Double label) {
			this.label = label;
		}

		public Double test(NumericalFeatureVector featureVector) {
			return label;
		}

		@Override
		public void train(List<NumericalFeatureVector> featureVectors, List<Double> labels) {
			for (int i = 0; i < featureVectors.size(); i++)
				if (labels.size() > i)
					label = labels.get(i);
		}

		@Override
		public List<Double> test(List<NumericalFeatureVector> featureVectors) {
			List<Double> labels = new ArrayList<>();

			for (int i = 0; i < featureVectors.size(); i++)
				if (labels.size() > i)
					labels.add(label);

			return labels;
		}

		@Override
		public Double getAccuracy(NumericalFeatureVector featureVector) {
			return 1.0;
		}
	}

	@SuppressWarnings("unchecked")
	@Test
	public void test() {
		@SuppressWarnings("rawtypes")
		ILearningModel model = new MyNum2NumLearningModel();

		List<NumericalFeatureVector> fvs = new ArrayList<>();
		fvs.add(null);

		List<Double> labels = new ArrayList<>();
		labels.add(2.0);

		model.train(fvs, labels);
		assert (model.test(null).equals(2.0));
	}
}
