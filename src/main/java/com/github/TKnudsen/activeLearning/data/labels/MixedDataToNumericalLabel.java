package com.github.TKnudsen.activeLearning.data.labels;

import java.util.AbstractMap;
import java.util.Map.Entry;

import com.github.TKnudsen.ComplexDataObject.data.features.mixedData.MixedDataFeatureVector;

/**
 * <p>
 * Title: MixedDataToNumericalLabel
 * </p>
 * 
 * <p>
 * Description: A numerical label for a MixedDataFeatureVector.
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
public class MixedDataToNumericalLabel implements ILabelObject<Object, MixedDataFeatureVector, Double> {

	Entry<MixedDataFeatureVector, Double> entry;

	public MixedDataToNumericalLabel(MixedDataFeatureVector featureVector, Double label) {
		this.entry = new AbstractMap.SimpleEntry<MixedDataFeatureVector, Double>(featureVector, label);
	}

	public MixedDataFeatureVector getFeatureVector() {
		return entry.getKey();
	}

	public Double getLabel() {
		return entry.getValue();
	}

}
