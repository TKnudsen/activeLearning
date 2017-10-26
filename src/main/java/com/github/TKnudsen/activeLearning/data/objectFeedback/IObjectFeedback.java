package com.github.TKnudsen.activeLearning.data.objectFeedback;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;

/**
 * <p>
 * Title: IObjectFeedback
 * </p>
 * 
 * <p>
 * Description: data structure that represents the way how a single object is
 * mapped in the feedback UI. Builds the basis for the extraction of labels.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.02
 */
public interface IObjectFeedback<O extends IDObject, FB extends Object> extends IObjectFeedbackProvider<O, FB> {

	public O getObject();

	public FB getFeedback();
}
