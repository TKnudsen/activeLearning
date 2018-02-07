package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import java.util.HashMap;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.IFeatureVectorObject;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.ComplexDataObject.model.statistics.Entropy;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

/**
 * <p>
 * Title: EntropyBasedActiveLearning
 * </p>
 * 
 * <p>
 * Description:
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016-2018 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.05
 */
public class EntropyBasedActiveLearning<FV extends IFeatureVectorObject<?, ?>> extends AbstractActiveLearningModel<FV> {
	protected EntropyBasedActiveLearning() {
	}

	@Deprecated
	public EntropyBasedActiveLearning(Classifier<FV> learningModel) {
		super(learningModel);
	}

	public EntropyBasedActiveLearning(IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier) {
		super(classificationResultSupplier);
	}

	@Override
	protected void calculateRanking(int count) {
		IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier = getClassificationResultSupplier();
		if (classificationResultSupplier == null)
			learningModel.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		queryApplicabilities = new HashMap<>();
		remainingUncertainty = 0.0;

		// calculate ranking based on entropy
		for (FV fv : learningCandidateFeatureVectors) {
			Map<String, Double> labelDistribution = null;
			if (classificationResultSupplier == null)
				labelDistribution = learningModel.getLabelDistribution(fv);
			else
				labelDistribution = classificationResultSupplier.get().getLabelDistribution(fv).getValueDistribution();

			double entropy = Entropy.calculateEntropy(labelDistribution);

			ranking.add(new EntryWithComparableKey<Double, FV>(1 - entropy, fv));

			queryApplicabilities.put(fv, entropy);
			remainingUncertainty += (entropy);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("EntropyBasedActiveLearning: remaining uncertainty = " + remainingUncertainty);
	}

	@Override
	public String getName() {
		return "Entropy-Based Sampling";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}
