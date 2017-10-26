package com.github.TKnudsen.activeLearning.data.objectFeedback;

import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;

/**
 * <p>
 * Title: IObjectsFeedback
 * </p>
 * 
 * <p>
 * Description: data structure reflecting the way how multiple objects are
 * represented/mapped in the feedback UI. Builds the basis for the extraction of
 * labels.
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
public interface IObjectsFeedback<O extends IDObject, FB extends Object> extends IObjectFeedbackProvider<O, FB> {

	public Map<O, FB> getFeedback();
}
