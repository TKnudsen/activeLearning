package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.IKeyValueProvider;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.ISelfDescription;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.model.supervised.ILearningModel;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.IClassifier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.IProbabilisticClassifier;

public abstract class AbstractActiveLearningModel<FV extends IKeyValueProvider<Object>>
		implements IActiveLearningModelClassification<FV>, ISelfDescription {

	protected List<FV> learningCandidateFeatureVectors;

	@Deprecated
	protected IProbabilisticClassifier<FV> learningModel;
	private IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier;

	protected Ranking<EntryWithComparableKey<Double, FV>> ranking;
	protected Map<FV, Double> queryApplicabilities;

	protected Double remainingUncertainty;

	protected AbstractActiveLearningModel() {

	}

	@Deprecated
	public AbstractActiveLearningModel(Classifier<FV> learningModel) {
		this.learningModel = learningModel;
	}

	public AbstractActiveLearningModel(IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier) {
		this.classificationResultSupplier = classificationResultSupplier;
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

	/**
	 * getRanking should not be used to get the next candidates. Use
	 * {@link AbstractActiveLearningModel#suggestCandidate()} or
	 * {@link AbstractActiveLearningModel#suggestCandidates(int)} instead.
	 * 
	 * @return
	 */
	@Deprecated
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
		if (queryApplicabilities == null && ranking != null)
			createQAfromRanking();
		if (queryApplicabilities != null)
			return queryApplicabilities.get(featureVector);

		return Double.NaN;
	}

	private void createQAfromRanking() {
		queryApplicabilities = new HashMap<>();
		for (EntryWithComparableKey<Double, FV> entry : ranking) {
			queryApplicabilities.put(entry.getValue(), -entry.getKey());
		}
	}

	/**
	 * copy of the applicability scores. high means applicable for AL.
	 * 
	 * @return
	 */
	public Map<FV, Double> getCandidateScores() {
		if (queryApplicabilities == null && ranking != null)
			createQAfromRanking();
		if (queryApplicabilities != null)
			return new LinkedHashMap<>(queryApplicabilities);

		return null;
	}

	@Override
	public double getRemainingUncertainty() {
		return remainingUncertainty;
	}

	@Override
	public ILearningModel<FV, String> getLearningModel() {
		return learningModel;
	}

	public void setLearningModel(IProbabilisticClassifier<FV> learningModel) {
		this.learningModel = learningModel;
	}

	@Override
	public String toString() {
		return this.getName() + " (" + this.learningModel.getName() + ")";
	}

	@Override
	public IProbabilisticClassificationResultSupplier<FV> getClassificationResultSupplier() {
		return classificationResultSupplier;
	}

	public void setClassificationResultSupplier(
			IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier) {
		this.classificationResultSupplier = classificationResultSupplier;
	}
}