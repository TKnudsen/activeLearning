package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.ISelfDescription;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

public abstract class AbstractActiveLearningModel<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> implements IActiveLearningModelClassification<O, FV>, ISelfDescription {

	protected List<FV> trainingFeatureVectors;
	protected List<FV> learningCandidateFeatureVectors;
	protected IClassifier<O, FV> learningModel;

	public AbstractActiveLearningModel(IClassifier<O, FV> learningModel) {
		this.learningModel = learningModel;
	}

	protected Ranking<EntryWithComparableKey<Double, FV>> ranking;
	protected Map<FV, Double> queryApplicabilities;

	protected Double remainingUncertainty;

	@Override
	public FV suggestCandidate() {
		List<FV> candidates = suggestCandidates(1);
		if (candidates != null && candidates.size() > 0)
			return candidates.get(0);

		return null;
	}

	@Override
	public List<FV> suggestCandidates(int count) {

		if (ranking == null || count > ranking.size())
			calculateRanking(count);

		List<FV> fvs = new ArrayList<>();
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
	public List<FV> getTrainingData() {
		return this.trainingFeatureVectors;
	}

	public Ranking<EntryWithComparableKey<Double, FV>> getRanking() {
		return ranking;
	}

	@Override
	public void setTrainingData(List<FV> featureVectors) {
		this.trainingFeatureVectors = featureVectors;

		ranking = null;
		queryApplicabilities = null;
	}

	public List<FV> getLearningCandidates() {

		return this.learningCandidateFeatureVectors;
	}

	@Override
	public void setLearningCandidates(List<FV> featureVectors) {
		this.learningCandidateFeatureVectors = featureVectors;

		ranking = null;
		queryApplicabilities = null;
	}

	@Override
	public double getCandidateApplicabilityScore(FV featureVector) {

		if (queryApplicabilities != null)
			return queryApplicabilities.get(featureVector);
		return Double.NaN;
	}

	@Override
	public void addCandidateVectorToTrainingVector(FV fv) {
		ranking = null;
		queryApplicabilities = null;

		if (this.learningCandidateFeatureVectors.contains(fv)) {
			this.learningCandidateFeatureVectors.remove(fv);
			this.trainingFeatureVectors.add(fv);
		} else
			throw new IllegalArgumentException("EvaluationBench.addCandidateVectorToTrainingVector: no such candidate vector.");
	}

	@Override
	public double getRemainingUncertainty() {
		return remainingUncertainty;
	}

	@Override
	public ILearningModel<O, FV, String> getLearningModel() {
		return learningModel;
	}

	@Override
	public String toString() {
		return this.getName() + " (" + this.learningModel.getName() + ")";
	}
}
