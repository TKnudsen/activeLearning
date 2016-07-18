package com.github.TKnudsen.activeLearning.models.learning;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;

/**
 * <p>
 * Title: INumericalToClassLearningModel
 * </p>
 * 
 * <p>
 * Description: basic algorithmic model that learns label information for
 * features. The labels are strings, thus, classifier-like models are trained.
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
public interface INumericalToClassLearningModel extends ILearningModel<Double, NumericalFeatureVector, String> {

}
