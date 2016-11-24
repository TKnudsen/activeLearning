package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * <p>
 * Title: DefaultActiveLearningModel
 * </p>
 * 
 * <p>
 * Description: the default active learning model combines two model-based
 * criteria for candidate selection: 1) candidates with very low class
 * affiliations 2) candidates with two competing class affiliations
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
public class DefaultActiveLearningModel implements IActiveLearningModelClassification<Double, NumericalFeatureVector> {

	private IClassifier<Double, NumericalFeatureVector> learningModel;

	public DefaultActiveLearningModel(IClassifier<Double, NumericalFeatureVector> learningModel) {
		this.learningModel = learningModel;
	}

	List<NumericalFeatureVector> trainingFeatureVectors;
	List<NumericalFeatureVector> learningCandidateFeatureVectors;

	private Map<NumericalFeatureVector, Double> relativeScoresMaxValue;
	private double relativeScoresMaxValueMin;
	private double relativeScoresMaxValueMax;
	private Map<NumericalFeatureVector, Double> relativeScoresDeltaMaxSecond;
	private double relativeScoresDeltaMaxSecondMin;
	private double relativeScoresDeltaMaxSecondMax;

	private Ranking<EntryWithComparableKey<Double, NumericalFeatureVector>> ranking;
	private Double remainingUncertainty;

	@Override
	public void setTrainingData(List<NumericalFeatureVector> featureVectors) {
		this.trainingFeatureVectors = featureVectors;
	}

	@Override
	public void setLearningCandidates(List<NumericalFeatureVector> featureVectors) {
		this.learningCandidateFeatureVectors = featureVectors;

		ranking = null;
	}

	private void refreshRelativeScores() {
		learningModel.test(learningCandidateFeatureVectors);

		relativeScoresMaxValue = new HashMap<>();
		relativeScoresDeltaMaxSecond = new HashMap<>();

		relativeScoresMaxValueMin = Double.POSITIVE_INFINITY;
		relativeScoresMaxValueMax = Double.NEGATIVE_INFINITY;
		relativeScoresDeltaMaxSecondMin = Double.POSITIVE_INFINITY;
		relativeScoresDeltaMaxSecondMax = Double.NEGATIVE_INFINITY;

		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			relativeScoresMaxValue.put(fv, learningModel.getLabelProbabilityMax(fv));
			relativeScoresMaxValueMin = Math.min(relativeScoresMaxValueMin, relativeScoresMaxValue.get(fv));
			relativeScoresMaxValueMax = Math.max(relativeScoresMaxValueMax, relativeScoresMaxValue.get(fv));

			relativeScoresDeltaMaxSecond.put(fv, learningModel.getLabelProbabilityDeltaMaxSecond(fv));
			relativeScoresDeltaMaxSecondMin = Math.min(relativeScoresDeltaMaxSecondMin, relativeScoresDeltaMaxSecond.get(fv));
			relativeScoresDeltaMaxSecondMax = Math.max(relativeScoresDeltaMaxSecondMax, relativeScoresDeltaMaxSecond.get(fv));
		}

		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			double v1 = relativeScoresMaxValue.get(fv);
			if (relativeScoresMaxValueMax == relativeScoresMaxValueMin)
				v1 = 0;
			else
				v1 = (v1 - relativeScoresMaxValueMin) / (relativeScoresMaxValueMax - relativeScoresMaxValueMin);
			relativeScoresMaxValue.put(fv, v1);

			double v2 = relativeScoresDeltaMaxSecond.get(fv);
			if (relativeScoresDeltaMaxSecondMax == relativeScoresDeltaMaxSecondMin)
				v2 = 0;
			else
				v2 = (v2 - relativeScoresDeltaMaxSecondMin) / (relativeScoresDeltaMaxSecondMax - relativeScoresDeltaMaxSecondMin);
			relativeScoresDeltaMaxSecond.put(fv, v2);
		}

		ranking = null;
	}

	private void calculateRanking(int count) {
		refreshRelativeScores();

		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			double v1 = relativeScoresMaxValue.get(fv);
			double v2 = relativeScoresDeltaMaxSecond.get(fv);
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(v1 + v2, fv));
			remainingUncertainty += (((1 - v1) + (1 - v2))*0.5);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
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

	@Override
	public ILearningModel<Double, NumericalFeatureVector, String> getLearningModel() {
		return learningModel;
	}

	@Override
	public double getRemainingUncertainty() {
		return remainingUncertainty;
	}
}
