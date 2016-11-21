package com.github.TKnudsen.activeLearning.models.evaluation;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;

public interface IModelEvaluation<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>, Y> {

	public double getQuality(ILearningModel<O, X, Y> model, List<X> testData, Y targetVariable);
}
