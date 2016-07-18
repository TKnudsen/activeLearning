package com.github.TKnudsen.activeLearning.data.objectFeedback.numerical;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;
import com.github.TKnudsen.activeLearning.data.objectFeedback.IObjectRelationFeedback;

/**
 * <p>
 * Title: ObjectRelationFeedbackNumerical
 * </p>
 * 
 * <p>
 * Description: reflects the positions of a number of objects aligned in 2D.
 * Builds the basis for the calculation of pairwise distances and thus for the
 * calculation of pairwise distance information.
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
public class ObjectRelationFeedbackNumerical<O extends IDObject> implements IObjectRelationFeedback<O, Double> {

	private O object1;
	private O object2;
	private Double feedback;

	public ObjectRelationFeedbackNumerical(O object1, O object2, Double feedback) {
		this.object1 = object1;
		this.object2 = object2;
		this.feedback = feedback;
	}

	public O getObject1() {
		return object1;
	}

	public O getObject2() {
		return object2;
	}

	public Double getFeedback() {
		return feedback;
	}

}
