package com.github.TKnudsen.activeLearning.data.objectFeedback.point;

import java.awt.geom.Point2D;
import java.util.HashMap;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IDObject;
import com.github.TKnudsen.activeLearning.data.objectFeedback.IObjectsFeedback;

/**
 * <p>
 * Title: ObjectsFeedbackPoints2D
 * </p>
 * 
 * <p>
 * Description: reflects the positions of a number of objects aligned in 2D.
 * Builds the basis for the calculation of pairwise distances and thus for the
 * calculation of pairwise distance information.
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
public class ObjectsFeedbackPoints2D<O extends IDObject> implements IObjectsFeedback<O, Point2D> {

	private Map<O, Point2D> geometryFeedback;

	public ObjectsFeedbackPoints2D() {
		geometryFeedback = new HashMap<O, Point2D>();
	}

	public ObjectsFeedbackPoints2D(Map<O, Point2D> geometryFeedback) {
		this.geometryFeedback = geometryFeedback;
	}

	public void addPoint(O object, Point2D point) {
		geometryFeedback.put(object, point);
	};

	public Map<O, Point2D> getFeedback() {
		return geometryFeedback;
	}

}
