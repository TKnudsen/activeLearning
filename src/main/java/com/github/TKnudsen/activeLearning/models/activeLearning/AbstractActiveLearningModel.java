package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.ISelfDescription;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.DMandML.model.supervised.ILearningModel;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;

public abstract class AbstractActiveLearningModel<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> implements IActiveLearningModelClassification<O, FV>, ISelfDescription {

	protected List<FV> learningCandidateFeatureVectors;

	protected Classifier<O, FV> learningModel;

	protected Ranking<EntryWithComparableKey<Double, FV>> ranking;
	protected Map<FV, Double> queryApplicabilities;

	protected Double remainingUncertainty;

	protected AbstractActiveLearningModel() {

	}

	public AbstractActiveLearningModel(Classifier<O, FV> learningModel) {
		this.learningModel = learningModel;
	}

	@Override
	public FV suggestCandidate() {
		List<FV> candidates = suggestCandidates(1);
		if (candidates != null && candidates.size() > 0)
			return candidates.get(0);

		return null;
	}

	@Override
	public List<FV> suggestCandidates(int count) {

		if (ranking == null || count > ranking.size()) {
			calculateRanking(count);
		}

		List<FV> fvs = new ArrayList<>();
		for (int i = 0; i < count; i++)
			fvs.add(i, ranking.get(i).getValue());

		return fvs;
	}

	protected abstract void calculateRanking(int count);

	public Ranking<EntryWithComparableKey<Double, FV>> getRanking() {
		return ranking;
	}

	public List<FV> getLearningCandidates() {
		return this.learningCandidateFeatureVectors;
	}

	@Override
	public void setLearningCandidates(List<FV> featureVectors) {
		this.learningCandidateFeatureVectors = featureVectors;

		clearResults();
	}

	public void clearResults() {
		ranking = null;
		queryApplicabilities = null;
	}

	@Override
	public double getCandidateApplicabilityScore(FV featureVector) {
		if (queryApplicabilities != null)
			return queryApplicabilities.get(featureVector);

		return Double.NaN;
	}

	/**
	 * copy of the applicability scores. high means applicable for AL.
	 * 
	 * @return
	 */
	public Map<FV, Double> getCandidateScores() {
		if (queryApplicabilities != null)
			return new LinkedHashMap<>(queryApplicabilities);

		return null;
	}

	@Override
	public double getRemainingUncertainty() {
		return remainingUncertainty;
	}

	@Override
	public ILearningModel<O, FV, String> getLearningModel() {
		return learningModel;
	}

	public void setLearningModel(Classifier<O, FV> learningModel) {
		this.learningModel = learningModel;
	}

	@Override
	public String toString() {
		return this.getName() + " (" + this.learningModel.getName() + ")";
	}
}