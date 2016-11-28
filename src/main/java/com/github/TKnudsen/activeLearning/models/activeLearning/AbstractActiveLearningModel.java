package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

public abstract class AbstractActiveLearningModel implements IActiveLearningModelClassification<Double, NumericalFeatureVector> {

	public AbstractActiveLearningModel(IClassifier<Double, NumericalFeatureVector> learningModel) {
		this.learningModel = learningModel;
	}

	protected List<NumericalFeatureVector> trainingFeatureVectors;
	protected List<NumericalFeatureVector> learningCandidateFeatureVectors;

	protected Ranking<EntryWithComparableKey<Double, NumericalFeatureVector>> ranking;
	protected Double remainingUncertainty;

	protected IClassifier<Double, NumericalFeatureVector> learningModel;

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
	public double getRemainingUncertainty() {
		return remainingUncertainty;
	}

	@Override
	public ILearningModel<Double, NumericalFeatureVector, String> getLearningModel() {
		return learningModel;
	}
}
