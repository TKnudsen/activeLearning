package com.github.TKnudsen.activeLearning.models.activeLearning;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IFeatureVectorObject;
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
 * Copyright: (c) 2016-2018 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.03
 */
public interface IActiveLearningModelClassification<FV extends IFeatureVectorObject<?, ?>>
		extends IActiveLearningModel<FV, String> {

	@Deprecated
	public ILearningModel<FV, String> getLearningModel();

	public IProbabilisticClassificationResultSupplier<FV> getClassificationResultSupplier();
}