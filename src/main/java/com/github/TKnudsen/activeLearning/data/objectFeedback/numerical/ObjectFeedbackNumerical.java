package com.github.TKnudsen.activeLearning.data.objectFeedback.numerical;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;
import com.github.TKnudsen.activeLearning.data.objectFeedback.IObjectFeedback;

/**
 * <p>
 * Title: ObjectFeedbackNumerical
 * </p>
 * 
 * <p>
 * Description: reflects the numerical feedback information provided for a given
 * IBObject.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.0
 */
public class ObjectFeedbackNumerical<O extends IDObject> implements IObjectFeedback<O, Double> {

	private O object;

	private Double feedback;

	public ObjectFeedbackNumerical(O object, Double feedback) {
		this.object = object;
		this.feedback = feedback;
	}

	public O getObject() {
		return object;
	}

	public Double getFeedback() {
		return feedback;
	}

}
