package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.ArrayList;
import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

public class SimpsonsDiversityActiveLearningModel implements IActiveLearningModelClassification<Double, NumericalFeatureVector> {

	List<NumericalFeatureVector> trainingFeatureVectors;
	List<NumericalFeatureVector> learningCandidateFeatureVectors;

	private Ranking<EntryWithComparableKey<Double, NumericalFeatureVector>> ranking;
	private Double remainingUncertainty;

	private IClassifier<Double, NumericalFeatureVector> learningModel;

	public SimpsonsDiversityActiveLearningModel(IClassifier<Double, NumericalFeatureVector> learningModel) {
		this.learningModel = learningModel;
	}

	@Override
	public void setTrainingData(List<NumericalFeatureVector> featureVectors) {
		this.trainingFeatureVectors = featureVectors;
	}

	@Override
	public void setLearningCandidates(List<NumericalFeatureVector> featureVectors) {
		this.learningCandidateFeatureVectors = featureVectors;

		ranking = null;
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

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			double v1 = learningModel.getLabelProbabilityDiversity(fv);
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(v1, fv));
			remainingUncertainty += (1 - v1);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
	}

	@Override
	public double getRemainingUncertainty() {
		return remainingUncertainty;
	}

	@Override
	public ILearningModel<Double, NumericalFeatureVector, String> getLearningModel() {
		return learningModel;
	}

}
