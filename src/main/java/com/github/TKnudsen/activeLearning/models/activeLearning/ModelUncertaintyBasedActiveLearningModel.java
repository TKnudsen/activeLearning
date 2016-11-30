package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.HashMap;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * <p>
 * Title: ModelUncertaintyBasedActiveLearningModel
 * </p>
 * 
 * <p>
 * Description: compares maximum values of class distributions for all feature
 * vectors. the candidate suggestion picks the set of weakest maximum values.
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
public class ModelUncertaintyBasedActiveLearningModel extends AbstractActiveLearningModel {

	public ModelUncertaintyBasedActiveLearningModel(IClassifier<Double, NumericalFeatureVector> learningModel) {
		super(learningModel);
	}

	private Map<NumericalFeatureVector, Double> scoresMaxValues;

	private void refreshRelativeScores() {
		learningModel.test(learningCandidateFeatureVectors);

		scoresMaxValues = new HashMap<>();
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors)
			scoresMaxValues.put(fv, learningModel.getLabelProbabilityMax(fv));

		ranking = null;
	}

	@Override
	protected void calculateRanking(int count) {
		refreshRelativeScores();

		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(scoresMaxValues.get(fv), fv));
			remainingUncertainty += (1 - scoresMaxValues.get(fv));

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("ModelUncertaintyBasedActiveLearningModel: remaining uncertainty = " + remainingUncertainty);
	}
}
