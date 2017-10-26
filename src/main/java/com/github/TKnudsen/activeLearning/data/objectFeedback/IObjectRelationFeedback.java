package com.github.TKnudsen.activeLearning.data.objectFeedback;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;

/**
 * <p>
 * Title: IObjectRelationFeedback
 * </p>
 * 
 * <p>
 * Description: data structure reflecting the way how a relation between two
 * objects is represented/mapped in the feedback UI. Builds the basis for the
 * extraction of labels.
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
public interface IObjectRelationFeedback<O extends IDObject, FB extends Object> extends IObjectFeedbackProvider<O, FB> {

	public O getObject1();

	public O getObject2();

	public FB getFeedback();

}
