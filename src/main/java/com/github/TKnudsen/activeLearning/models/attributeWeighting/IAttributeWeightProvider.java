package com.github.TKnudsen.activeLearning.models.attributeWeighting;

/**
 * <p>
 * Title: IAttributeWeightProvider
 * </p>
 * 
 * <p>
 * Description: provides weight information for an attribute.
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
public interface IAttributeWeightProvider {

	public Double getWeight(String attribute);
}
