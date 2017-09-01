package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import java.util.HashMap;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

public class EntropyBasedActiveLearning<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	public EntropyBasedActiveLearning(Classifier<O, FV> learningModel) {
		super(learningModel);
	}

	@Override
	protected void calculateRanking(int count) {
		learningModel.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		queryApplicabilities = new HashMap<>();
		remainingUncertainty = 0.0;

		// calculate ranking based on entropy
		for (FV fv : learningCandidateFeatureVectors) {
			Map<String, Double> distribution = learningModel.getLabelDistribution(fv);
			// System.out.println(distribution);

			double entropy = calculateEntropy(distribution);
			// System.out.println(entropy);

			ranking.add(new EntryWithComparableKey<Double, FV>(1 - entropy, fv));

			queryApplicabilities.put(fv, entropy);
			remainingUncertainty += (entropy);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("EntropyBasedActiveLearning: remaining uncertainty = " + remainingUncertainty);
	}

	public static double calculateEntropy(Map<String, Double> distribution) {
		if (distribution == null || distribution.size() == 0)
			return 0;

		double entropy = 0.0;
		for (String s : distribution.keySet())
			if (distribution.get(s) > 0)
				entropy -= (distribution.get(s) * Math.log(distribution.get(s)));

		entropy /= Math.log(2.0);

		return entropy;
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
