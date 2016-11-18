package com.github.TKnudsen.activeLearning.models.learning;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;

/**
 * <p>
 * Title: ILearningModel
 * </p>
 * 
 * <p>
 * Description: basic algorithmic model that learns label information for
 * features.
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
public interface ILearningModel<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>, Y> {

	public void train(List<X> featureVectors, List<Y> labels);

	public void train(List<X> featureVectors, String targetVariable);

	public List<Y> test(List<X> featureVectors);
}
