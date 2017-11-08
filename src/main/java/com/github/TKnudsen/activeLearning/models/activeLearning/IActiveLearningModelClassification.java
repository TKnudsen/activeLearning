package com.github.TKnudsen.activeLearning.models.activeLearning;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.model.supervised.ILearningModel;

/**
 * <p>
 * Title: IActiveLearningModelClassification
 * </p>
 * 
 * <p>
 * Description: algorithmic model that estimates the coverage of the feature
 * space with respect to already labeled features.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.02
 */
public interface IActiveLearningModelClassification<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>> extends IActiveLearningModel<O, X, String> {

	@Deprecated
	public ILearningModel<O, X, String> getLearningModel();

	public IProbabilisticClassificationResultSupplier<X> getClassificationResultSupplier();
}