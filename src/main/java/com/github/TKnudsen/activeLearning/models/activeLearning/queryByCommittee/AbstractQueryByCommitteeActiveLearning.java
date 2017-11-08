package com.github.TKnudsen.activeLearning.models.activeLearning.queryByCommittee;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.model.supervised.ILearningModel;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

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
 * Copyright: (c) 2016-2017 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 *
 * @author Juergen Bernard
 * @version 1.03
 */
public abstract class AbstractQueryByCommitteeActiveLearning<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	@Deprecated
	private List<Classifier<O, FV>> learningModels;

	private List<IProbabilisticClassificationResultSupplier<FV>> classificationResultSuppliers;

	protected AbstractQueryByCommitteeActiveLearning() {
	}

	@Deprecated
	public AbstractQueryByCommitteeActiveLearning(List<Classifier<O, FV>> learningModels) {
		super(learningModels.get(0));
		this.learningModels = learningModels;
	}

	public AbstractQueryByCommitteeActiveLearning(List<IProbabilisticClassificationResultSupplier<FV>> classificationResultSuppliers, boolean fakeBooleanToBeDifferentThanDeprecateConstructor) {
		super(classificationResultSuppliers.get(0));

		this.classificationResultSuppliers = classificationResultSuppliers;
	}

	public abstract String getComparisonMethod();

	@Override
	public ILearningModel<O, FV, String> getLearningModel() {
		if (learningModels != null && learningModels.size() > 0)
			return learningModels.get(0);

		return null;
	}

	@Deprecated
	public List<Classifier<O, FV>> getLearningModels() {
		return learningModels;
	}

	@Deprecated
	public void setLearningModels(List<Classifier<O, FV>> learningModels) {
		this.learningModels = learningModels;
	}

	public List<IProbabilisticClassificationResultSupplier<FV>> getClassificationResultSuppliers() {
		return classificationResultSuppliers;
	}

	public void setClassificationResultSuppliers(List<IProbabilisticClassificationResultSupplier<FV>> classificationResultSuppliers) {
		this.classificationResultSuppliers = classificationResultSuppliers;
	}
}
