package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import java.util.List;

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
	public List<NumericalFeatureVector> suggestCandidates(int count) {
		return null;
	}

	@Override
	protected void calculateRanking(int count) {
		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

//		learningModel.test(learningCandidateFeatureVectors);

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			double v1 = learningModel.getLabelProbabilityMax(fv);
			double v2 = learningModel.getLabelProbabilityDeltaMaxSecond(fv);
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(Math.abs(v1 - v2), fv));
			remainingUncertainty += (1 - Math.abs(v1 - v2));

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
