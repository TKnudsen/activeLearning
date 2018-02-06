package com.github.TKnudsen.activeLearning.models.activeLearning.queryByCommittee;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.interfaces.IFeatureVectorObject;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;

/**
 * <p>
 * Title: VoteComparisonQueryByCommittee
 * </p>
 * 
 * <p>
 * Description: queries controversial instances/regions in the input space.
 * Compares the label distributions of every candidate for a given set of
 * models. The winning candidate poses those label distributions where the
 * committee disagrees most.
 * 
 * Measure: Vote Comparison. Ratio of different Votes.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016-2018 Juergen Bernard
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.04
 */
public class VoteComparisonQueryByCommittee<FV extends IFeatureVectorObject<?, Feature<?>>>
		extends AbstractQueryByCommitteeActiveLearning<FV> {
	protected VoteComparisonQueryByCommittee() {
	}

	public VoteComparisonQueryByCommittee(List<Classifier<FV>> learningModels) {
		super(learningModels);
	}

	public VoteComparisonQueryByCommittee(
			List<IProbabilisticClassificationResultSupplier<FV>> classificationResultSuppliers,
			boolean fakeBooleanToBeDifferentThanDeprecateConstructor) {
		super(classificationResultSuppliers, false);
	}

	@Override
	public String getComparisonMethod() {
		return "Measures the ratio of disagreeing suggestions for labels of different models (votes).";
	}

	@Override
	protected void calculateRanking(int count) {
		List<IProbabilisticClassificationResultSupplier<FV>> classificationResultSuppliers = getClassificationResultSuppliers();

		if (classificationResultSuppliers == null || classificationResultSuppliers.size() == 0)
			for (Classifier<FV> classifier : getLearningModels())
				classifier.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		queryApplicabilities = new HashMap<>();
		remainingUncertainty = 0.0;

		// calculate overall score
		for (FV fv : learningCandidateFeatureVectors) {
			List<Map<String, Double>> labelDistributions = new ArrayList<>();
			if (classificationResultSuppliers == null || classificationResultSuppliers.size() == 0)
				for (Classifier<FV> classifier : getLearningModels())
					labelDistributions.add(classifier.getLabelDistribution(fv));
			else
				for (IProbabilisticClassificationResultSupplier<FV> result : classificationResultSuppliers)
					labelDistributions.add(result.get().getLabelDistribution(fv).getValueDistribution());

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

			if (distributions != null && distributions.size() > 0) {
				Set<String> winningLabels = new HashSet<>();
				if (classificationResultSuppliers == null || classificationResultSuppliers.size() == 0)
					for (Classifier<FV> classifier : getLearningModels()) {
						List<String> test = classifier.test(Arrays.asList(fv));
						if (test != null && test.size() > 0)
							winningLabels.add(classifier.test(Arrays.asList(fv)).get(0));
					}
				else
					for (IProbabilisticClassificationResultSupplier<FV> result : classificationResultSuppliers) {
						winningLabels.add(result.get().getLabelDistribution(fv).getRepresentant());

					}

				dist = (winningLabels.size() - 1) / (double) distributions.size();
			} else
				dist = 1;

			// update ranking

			ranking.add(new EntryWithComparableKey<Double, FV>(1 - dist, fv));

			queryApplicabilities.put(fv, dist);
			remainingUncertainty += dist;

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("VoteComparisonQueryByCommittee: remaining uncertainty = " + remainingUncertainty);
	}

	@Override
	public String getName() {
		return "Vote Comparison QBC";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}
