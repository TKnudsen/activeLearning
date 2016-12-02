package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * <p>
 * Title: LastSignificantConfidence
 * </p>
 * 
 * <p>
 * Description: a baseline active learning model seeking the smallest difference
 * between the first and second most probable class labels among all instances.
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
public class SmallestMarginActiveLearning extends AbstractActiveLearningModel {

	public SmallestMarginActiveLearning(IClassifier<Double, NumericalFeatureVector> learningModel) {
		super(learningModel);
	}

	@Override
	protected void calculateRanking(int count) {
		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

		// learningModel.test(learningCandidateFeatureVectors);

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			double margin = learningModel.getLabelProbabilityMargin(fv);
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(margin, fv));
			remainingUncertainty += (1 - margin);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("SmallestMarginActiveLearning: remaining uncertainty = " + remainingUncertainty);
	}

	@Override
	public String getName() {
		return "Smallest MMargin";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}
