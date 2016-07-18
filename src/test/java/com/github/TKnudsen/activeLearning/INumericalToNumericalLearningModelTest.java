package com.github.TKnudsen.activeLearning;

import org.junit.Test;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.INumericalToNumericalLearningModel;

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

	static class MyNum2NumLearningModel implements INumericalToNumericalLearningModel {
		Double label;

		public void train(NumericalFeatureVector featureVector, Double label) {
			this.label = label;
		}

		public Double test(NumericalFeatureVector featureVector) {
			// TODO Auto-generated method stub
			return label;
		}

	}

	@SuppressWarnings("unchecked")
	@Test
	public void test() {
		@SuppressWarnings("rawtypes")
		ILearningModel model = new MyNum2NumLearningModel();

		model.train(null, 2.0);
		assert (model.test(null).equals(2.0));
	}

}
