package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.ArrayList;
import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.ISelfDescription;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

public abstract class AbstractActiveLearningModel implements IActiveLearningModelClassification<Double, NumericalFeatureVector>, ISelfDescription {

	public AbstractActiveLearningModel(IClassifier<Double, NumericalFeatureVector> learningModel) {
		this.learningModel = learningModel;
	}
	
	protected List<NumericalFeatureVector> trainingFeatureVectors;
	protected List<NumericalFeatureVector> learningCandidateFeatureVectors;

	protected Ranking<EntryWithComparableKey<Double, NumericalFeatureVector>> ranking;
	protected Double remainingUncertainty;

	protected IClassifier<Double, NumericalFeatureVector> learningModel;

	@Override
	public List<NumericalFeatureVector> suggestCandidates(int count) {

		if (ranking == null)
			calculateRanking(count);

		List<NumericalFeatureVector> fvs = new ArrayList<>();
		for (int i = 0; i < ranking.size(); i++)
			fvs.add(i, ranking.get(i).getValue());

		return fvs;
	}

	protected abstract void calculateRanking(int count);

	/**
	 * information about training data is not necessarily relevant for AL
	 * models. Nevertheless we provide the information for convenience reasons.
	 * 
	 * @return
	 */
	public List<NumericalFeatureVector> getTrainingData() {
		return this.trainingFeatureVectors;
	}

	@Override
	public void setTrainingData(List<NumericalFeatureVector> featureVectors) {
		this.trainingFeatureVectors = featureVectors;
	}

	public List<NumericalFeatureVector> getLearningCandidates() {
		return this.learningCandidateFeatureVectors;
	}

	@Override
	public void setLearningCandidates(List<NumericalFeatureVector> featureVectors) {
		this.learningCandidateFeatureVectors = featureVectors;

		ranking = null;
	}

	@Override
	public void addCandidateVectorToTrainingVector(NumericalFeatureVector fv) {
		if (this.learningCandidateFeatureVectors.contains(fv)) {
			this.learningCandidateFeatureVectors.remove(fv);
			this.trainingFeatureVectors.add(fv);
		} else
			throw new IllegalArgumentException("EvaluationBench.addCandidateVectorToTrainingVector: no such candidate vector.");

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
