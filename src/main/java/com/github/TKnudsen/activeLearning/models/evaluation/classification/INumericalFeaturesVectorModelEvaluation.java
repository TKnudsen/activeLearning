package com.github.TKnudsen.activeLearning.models.evaluation.classification;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.activeLearning.models.evaluation.IModelEvaluation;

public interface INumericalFeaturesVectorModelEvaluation<Y> extends IModelEvaluation<Double, NumericalFeatureVector, Y> {

}
