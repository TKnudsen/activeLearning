package com.github.TKnudsen.activeLearning.models.activeLearning.queryByCommittee;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.activeLearning.IActiveLearningModelClassification;
import com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling.EntropyBasedActiveLearning;
import com.github.TKnudsen.activeLearning.models.learning.ILearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * <p>
 * Title: QueryByCommitteeActiveLearningModel
 * </p>
 * 
 * <p>
 * Description: queries controversial instances/regions in the input space.
 * Compares the label distributions of every candidate for a given set of
 * models. The winning candidate poses those label distributions where the
 * committee disagrees most.
 * 
 * Degree of freedom: measure of disagreement among committee members
 * represented with the enum ComparisonMethod.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.02
 */
public class QueryByCommitteeActiveLearningModel implements IActiveLearningModelClassification<Double, NumericalFeatureVector> {

	private List<IClassifier<Double, NumericalFeatureVector>> learningModels;

	public enum ComparisonMethod {
		DistanceBased, MaximumComparison, Entropy
	};

	private ComparisonMethod comparisonMethod;

	public QueryByCommitteeActiveLearningModel(List<IClassifier<Double, NumericalFeatureVector>> learningModels) {
		this.learningModels = learningModels;
	}

	List<NumericalFeatureVector> trainingFeatureVectors;
	List<NumericalFeatureVector> learningCandidateFeatureVectors;

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

	private void calculateRanking(int count) {
		for (IClassifier<Double, NumericalFeatureVector> classifier : learningModels)
			classifier.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

		// calculate overall score
		for (NumericalFeatureVector fv : learningCandidateFeatureVectors) {
			List<Map<String, Double>> labelDistributions = new ArrayList<>();
			for (IClassifier<Double, NumericalFeatureVector> classifier : learningModels)
				labelDistributions.add(classifier.getLabelDistribution(fv));

			// create unified distribution arrays
			Set<String> labelSet = new HashSet<>();
			for (Map<String, Double> map : labelDistributions)
				if (map != null)
					for (String s : map.keySet())
						labelSet.add(s);

			List<List<Double>> distributions = new ArrayList<>();
			for (Map<String, Double> map : labelDistributions) {
				if (map == null)
					continue;
				List<Double> values = new ArrayList<>();
				for (String s : labelSet)
					if (map.get(s) != null)
						values.add(map.get(s));
					else
						values.add(0.0);
				distributions.add(values);
			}

			double dist = 0;
			switch (comparisonMethod) {
			case DistanceBased:
				// calculate pairwise distances
				for (int i = 0; i < distributions.size() - 1; i++)
					for (int j = i + 1; j < distributions.size(); j++)
						dist += calculateDistance(distributions.get(i), distributions.get(j));
				if (labelSet.size() > 0)
					dist /= ((distributions.size() - 1) * (distributions.size() + 1 - 1) * 0.5);
				else
					dist = 1;
				dist = (Math.max(0, Math.min(dist, 1)));
				break;
			case MaximumComparison:
				// compare winning labels
				if (distributions != null && distributions.size() > 0) {
					Set<String> winningLabels = new HashSet<>();
					for (IClassifier<Double, NumericalFeatureVector> classifier : learningModels) {
						List<String> test = classifier.test(Arrays.asList(fv));
						if (test != null && test.size() > 0)
							winningLabels.add(classifier.test(Arrays.asList(fv)).get(0));
					}
					dist = (winningLabels.size() - 1) / (double) distributions.size();
				} else
					dist = 1;
				// dist = (Math.max(0, Math.min(dist, 1)));
				break;
			case Entropy:
				// create distribution of winning labels
				if (distributions != null && distributions.size() > 0) {
					Map<String, Double> winningLabels = new HashMap();
					for (IClassifier<Double, NumericalFeatureVector> classifier : learningModels) {
						List<String> test = classifier.test(Arrays.asList(fv));
						String label = classifier.test(Arrays.asList(fv)).get(0);
						if (test != null && test.size() > 0)
							if (!winningLabels.containsKey(label))
								winningLabels.put(label, 1.0);
							else
								winningLabels.put(label, winningLabels.get(label) + 1.0);
					}

					for (String label : winningLabels.keySet())
						winningLabels.put(label, winningLabels.get(label) / (double) learningModels.size());

					dist = EntropyBasedActiveLearning.calculateEntropy(winningLabels);
				} else
					dist = 1;
				break;
			default:
				dist = 0;
				System.err.println("QueryByCommitteeActiveLearningModel: undefined comparison method");
				break;
			}

			// update ranking
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(1 - dist, fv));
			remainingUncertainty += dist;

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("QueryByCommitteeActiveLearningModel: remaining uncertainty = " + remainingUncertainty);
	}

	private double calculateDistance(List<Double> v1, List<Double> v2) {
		double d = 0;
		for (int i = 0; i < Math.min(v1.size(), v2.size()); i++)
			d += Math.pow(v1.get(i).doubleValue() - v2.get(i).doubleValue(), 2);
		d = Math.sqrt(d);

		return d;
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
		if (learningModels != null && learningModels.size() > 0)
			return learningModels.get(0);

		return null;
	}

	@Override
	public double getRemainingUncertainty() {
		return remainingUncertainty;
	}

	public ComparisonMethod getComparisonMethod() {
		return comparisonMethod;
	}

	public void setComparisonMethod(ComparisonMethod comparisonMethod) {
		this.comparisonMethod = comparisonMethod;
	}
}
