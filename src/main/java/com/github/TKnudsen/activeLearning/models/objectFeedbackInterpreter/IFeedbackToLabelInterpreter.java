package com.github.TKnudsen.activeLearning.models.objectFeedbackInterpreter;

import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;
import com.github.TKnudsen.activeLearning.data.objectFeedback.IObjectFeedbackProvider;

/**
 * <p>
 * Title: IFeedbackToLabelInterpreter
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
public interface IFeedbackToLabelInterpreter<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>, Y, IDO extends IDObject, FB extends Object> {

	public Map<X, Y> getLabels(IObjectFeedbackProvider<IDO, FB> objectFeedback);
}
