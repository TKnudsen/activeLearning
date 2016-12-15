package com.github.TKnudsen.activeLearning.models.learning.classification;

import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;

/**
 * <p>
 * Title: IClassifier
 * </p>
 * 
 * <p>
 * Description: basic algorithmic model that learns categorical label
 * information.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.03
 */
public interface IClassifier<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>> extends ILearningModel<O, X, String> {

	public Map<String, Double> getLabelDistribution(X featureVector);

	public double getLabelProbabilityMax(X featureVector);

	public double getLabelProbabilityMargin(X featureVector);

	public double getLabelProbabilityDiversity(X featureVector);

	/**
	 * Convenient functionality to create 'model clones' as individual instance
	 * with identical parameter sets. copies do not contain learning
	 * information.
	 * 
	 * @return
	 */
	public IClassifier<O, X> createParameterizedCopy() throws Exception;
}
