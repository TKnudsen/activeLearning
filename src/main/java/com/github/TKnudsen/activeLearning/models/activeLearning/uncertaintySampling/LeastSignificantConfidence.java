package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import java.util.HashMap;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * <p>
 * Title: LeastSignificantConfidence
 * </p>
 * 
 * <p>
 * Description: a baseline active learning model seeking the lowest maximum
 * likelihood among all instances.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.01
 */
public class LeastSignificantConfidence<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	public LeastSignificantConfidence(IClassifier<O, FV> learningModel) {
		super(learningModel);
	}

	@Override
	protected void calculateRanking(int count) {
		learningModel.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		queryApplicabilities = new HashMap<>();
		remainingUncertainty = 0.0;

		learningModel.test(learningCandidateFeatureVectors);

		// calculate overall score
		for (FV fv : learningCandidateFeatureVectors) {
			double likelihood = learningModel.getLabelProbabilityMax(fv);
			ranking.add(new EntryWithComparableKey<Double, FV>(likelihood, fv));
			queryApplicabilities.put(fv, 1 - likelihood);
			remainingUncertainty += (1 - likelihood);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("LastSignificantConfidence: remaining uncertainty = " + remainingUncertainty);
	}

	@Override
	public String getName() {
		return "Last Significant Confidence";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}

			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(likelihood, fv));
			queryApplicabilities.put(fv, 1 - likelihood);
			remainingUncertainty += (1 - likelihood);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("LastSignificantConfidence: remaining uncertainty = " + remainingUncertainty);
	}

	@Override
	public String getName() {
		return "Last Significant Confidence";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}
