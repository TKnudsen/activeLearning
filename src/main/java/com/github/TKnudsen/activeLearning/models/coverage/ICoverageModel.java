package com.github.TKnudsen.activeLearning.models.coverage;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;

/**
 * <p>
 * Title: ICoverageModel
 * </p>
 * 
 * <p>
 * Description: algorithmic model that estimates the coverage of the feature
 * space with respect to already labeled features.
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
public interface ICoverageModel<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>> {

	public void addLabeledCandidate(X featureVector);

	public Double getCoverageScore(X featureVector);

	public List<X> suggestCandidates();

	public double getRemainingUncertainty();

}
