package com.github.TKnudsen.activeLearning.models.objectFeedbackInterpreter.objects;

import java.awt.geom.Point2D;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;
import com.github.TKnudsen.activeLearning.data.objectFeedback.IObjectFeedbackProvider;
import com.github.TKnudsen.activeLearning.models.objectFeedbackInterpreter.IFeedbackToLabelInterpreter;

/**
 * <p>
 * Title: Point2DDistributionInterpreter
 * </p>
 * 
 * <p>
 * Description: algorithmic model that interprets user feedback
 * (IObjectFeedbackProvider) and produces label information.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.0
 */
public class Point2DDistributionInterpreter implements IFeedbackToLabelInterpreter<Double, NumericalFeatureVector, Double, IDObject, Point2D.Double> {

	public Map<NumericalFeatureVector, Double> getLabels(IObjectFeedbackProvider<IDObject, java.awt.geom.Point2D.Double> objectFeedback) {
		// TODO Transform point distribution to numerical label information.
		// TODO use delta FeatureVector to reflect pairwise relative distances
		return null;
	}

}
