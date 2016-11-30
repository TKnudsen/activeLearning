package com.github.TKnudsen.activeLearning.models.activeLearning.queryByCommittee;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * <p>
 * Title: AbstractQueryByCommitteeActiveLearning
 * </p>
 * 
 * <p>
 * Description: queries controversial instances/regions in the input space.
 * Compares the label distributions of every candidate for a given set of
 * models. The winning candidate poses those label distributions where the
 * committee disagrees most.
 * 
 * Degree of freedom: measure of disagreement among committee members. See the
 * inheriting classes.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.02
 */
public abstract class AbstractQueryByCommitteeActiveLearning extends AbstractActiveLearningModel {

	protected List<IClassifier<Double, NumericalFeatureVector>> learningModels;

	public AbstractQueryByCommitteeActiveLearning(List<IClassifier<Double, NumericalFeatureVector>> learningModels) {
		super(learningModels.get(0));
		this.learningModels = learningModels;
	}

	public abstract String getComparisonMethod();

	@Override
	public ILearningModel<Double, NumericalFeatureVector, String> getLearningModel() {
		if (learningModels != null && learningModels.size() > 0)
			return learningModels.get(0);

		return null;
	}
}
