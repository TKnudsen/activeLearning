package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
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
public class LeastSignificantConfidence extends AbstractActiveLearningModel {

	public LeastSignificantConfidence(IClassifier<Double, NumericalFeatureVector> learningModel) {
		super(learningModel);
	}

	@Override
	protected void calculateRanking(int count) {
		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

		learningModel.test(learningCandidateFeatureVectors);

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			double likelihood = learningModel.getLabelProbabilityMax(fv);
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(likelihood, fv));
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
