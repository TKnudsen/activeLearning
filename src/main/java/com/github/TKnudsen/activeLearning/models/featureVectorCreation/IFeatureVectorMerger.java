package com.github.TKnudsen.activeLearning.models.featureVectorCreation;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;

/**
 * <p>
 * Title: IFeatureVectorMerger
 * </p>
 * 
 * <p>
 * Description: algorithmic model that condenses the information of two
 * FeatureVectors into a new FeatureVector.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.01
 */
public interface IFeatureVectorMerger<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>> {

	public X createFeatureVector(X featureVector1, X featureVector2);
}
