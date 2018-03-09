package com.github.TKnudsen.activeLearning.models.activeLearning.queryByCommittee;

import java.util.List;

import org.apache.commons.lang3.NotImplementedException;

import com.github.TKnudsen.ComplexDataObject.data.interfaces.IFeatureVectorObject;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResult;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.data.classification.LabelDistribution;

/**
 * <p>
 * Title: ActiveDecorateQueryByCommittee
 * </p>
 * 
 * <p>
 * Description: constructs diverse committees using artificial training data.
 * Builds upon decorate committees.
 * 
 * Measure: Kullback-Leibler Divergence. Divergence between models' label
 * probability distribution and consensus distribution.
 * 
 * Citation: Diverse Ensembles for Active Learning. Prem Melville and Raymond J.
 * Mooney. Machine Learning, 2004.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016-2018 Juergen Bernard
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.01
 */

public class ActiveDecorateQueryByCommittee<FV extends IFeatureVectorObject<?, ?>>
		extends AbstractQueryByCommitteeActiveLearning<FV> {

	protected ActiveDecorateQueryByCommittee() {
	}

	public ActiveDecorateQueryByCommittee(
			List<IProbabilisticClassificationResultSupplier<FV>> classificationResultSuppliers,
			boolean fakeBooleanToBeDifferentThanDeprecateConstructor) {
		super(classificationResultSuppliers, false);
	}

	@Override
	public String getComparisonMethod() {
		return "Detects disagreement among an ensemble of hypotheses (classifiers)";
	}

	@Override
	protected void calculateRanking(int count) {

		// TODO use the classification results to identify instances with disagreement.
		List<IProbabilisticClassificationResultSupplier<FV>> classificationResultSuppliers = getClassificationResultSuppliers();

		for (IProbabilisticClassificationResultSupplier<FV> crs : classificationResultSuppliers) {
			IProbabilisticClassificationResult<FV> classificationResult = crs.get();

			for (FV fv : learningCandidateFeatureVectors) {
				LabelDistribution labelDistribution = classificationResult.getLabelDistribution(fv);

				// t.b.d.
			}
		}

		throw new NotImplementedException(getName() + ".getRanking not implemented yet");
	}

	@Override
	public String getName() {
		return "Active-Decorate QBC";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}
