package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.ArrayList;
import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

public class SimpsonsDiversityActiveLearningModel extends AbstractActiveLearningModel {

	public SimpsonsDiversityActiveLearningModel(IClassifier<Double, NumericalFeatureVector> learningModel) {
		super(learningModel);
	}

	@Override
	public List<NumericalFeatureVector> suggestCandidates(int count) {
		if (ranking == null)
			calculateRanking(count);

		List<NumericalFeatureVector> fvs = new ArrayList<>();
		for (int i = 0; i < ranking.size(); i++)
			fvs.add(i, ranking.get(i).getValue());

		return fvs;
	}

	private void calculateRanking(int count) {
		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

		// learningModel.test(learningCandidateFeatureVectors);

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			double v1 = learningModel.getLabelProbabilityDiversity(fv);
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(v1, fv));
			remainingUncertainty += (1 - v1);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("SimpsonsDiveristyActiveLearningModel: remaining uncertainty = " + remainingUncertainty);
	}

}
