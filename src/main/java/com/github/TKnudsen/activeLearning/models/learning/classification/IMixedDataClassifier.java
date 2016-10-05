package com.github.TKnudsen.activeLearning.models.learning.classification;

import com.github.TKnudsen.ComplexDataObject.data.features.mixedData.MixedDataFeatureVector;

/**
 * <p>
 * Title: IMixedDataToClassLearningModel
 * </p>
 * 
 * <p>
 * Description: basic algorithmic model that learns categorical label
 * information for mixed data features. The labels are strings, thus,
 * classifier-like models are trained.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.02
 */
public interface IMixedDataClassifier extends IClassifier<Object, MixedDataFeatureVector> {

}
