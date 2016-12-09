package com.github.TKnudsen.activeLearning.models.weighting;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;

public interface IFeatureVectorWeighting<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>> {

	public void setWeight(List<X> featureVectors, List<Double> weights);
}
