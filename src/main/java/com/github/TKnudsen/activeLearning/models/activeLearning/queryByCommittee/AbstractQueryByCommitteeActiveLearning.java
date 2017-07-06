package com.github.TKnudsen.activeLearning.models.activeLearning.queryByCommittee;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

import main.java.com.github.TKnudsen.DMandML.model.supervised.ILearningModel;
import main.java.com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;

/**
 * <p>
 * Title: AbstractQueryByCommitteeActiveLearning
 * </p>
 * <p>
 * <p>
 * Description: queries controversial instances/regions in the input space.
 * Compares the label distributions of every candidate for a given set of
 * models. The winning candidate poses those label distributions where the
 * committee disagrees most.
 * <p>
 * Degree of freedom: measure of disagreement among committee members. See the
 * inheriting classes.
 * </p>
 * <p>
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 *
 * @author Juergen Bernard
 * @version 1.02
 */
public abstract class AbstractQueryByCommitteeActiveLearning<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	protected List<Classifier<O, FV>> learningModels;

	public AbstractQueryByCommitteeActiveLearning(List<Classifier<O, FV>> learningModels) {
		super(learningModels.get(0));
		this.learningModels = learningModels;
	}

	public abstract String getComparisonMethod();

	@Override
	public ILearningModel<O, FV, String> getLearningModel() {
		if (learningModels != null && learningModels.size() > 0)
			return learningModels.get(0);

		return null;
	}
}
