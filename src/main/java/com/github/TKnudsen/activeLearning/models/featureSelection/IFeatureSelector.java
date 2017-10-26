package com.github.TKnudsen.activeLearning.models.featureSelection;

import java.util.List;
import java.util.Map.Entry;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;

/**
 * <p>
 * Title: IFeatureSelector
 * </p>
 * 
 * <p>
 * Description: algorithmic model that provides feature selection capability.
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
public interface IFeatureSelector<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>> {

	public String suggestFeatureAttributeToBeRemoved();

	public List<String> selectNumerOfMostRelevantFeatures(int number);

	public Entry<String, String> suggestFeatureAttributesToBeMerged();
}
