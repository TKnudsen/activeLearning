package com.github.TKnudsen.activeLearning.data.labels;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;

/**
 * <p>
 * Title: ILabelObject
 * </p>
 * 
 * <p>
 * Description: Provides the label information for a given FeatureVector
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.01
 */
public interface ILabelObject<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>, Y> {

	public X getFeatureVector();

	public Y getLabel();

}
