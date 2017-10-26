package com.github.TKnudsen.activeLearning.data.labels;

import java.util.AbstractMap;
import java.util.Map.Entry;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;

/**
 * <p>
 * Title: NumericalToNumericalLabel
 * </p>
 * 
 * <p>
 * Description: A numerical label for a numerical FeatureVector
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
public class NumericalToNumericalLabel implements ILabelObject<Double, NumericalFeatureVector, Double> {

	Entry<NumericalFeatureVector, Double> entry;

	public NumericalToNumericalLabel(NumericalFeatureVector numericalFeatureVector, Double label) {
		this.entry = new AbstractMap.SimpleEntry<NumericalFeatureVector, Double>(numericalFeatureVector, label);
	}

	public NumericalFeatureVector getFeatureVector() {
		return entry.getKey();
	}

	public Double getLabel() {
		return entry.getValue();
	}
}
