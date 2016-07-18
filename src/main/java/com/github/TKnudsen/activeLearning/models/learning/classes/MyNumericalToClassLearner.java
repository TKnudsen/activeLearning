package com.github.TKnudsen.activeLearning.models.learning.classes;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.activeLearning.models.learning.INumericalToClassLearningModel;

/**
 * <p>
 * Title: MyNumericalToClassLearner
 * </p>
 * 
 * <p>
 * Description: Dummy Classifier accepting numerical features.
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
public class MyNumericalToClassLearner implements INumericalToClassLearningModel {

	private String label;

	public void train(NumericalFeatureVector featureVector, String label) {
		this.label = label;

		// TODO include model
	}

	public String test(NumericalFeatureVector featureVector) {
		return label;
	}

}
