package com.github.TKnudsen.activeLearning.models.activeLearning.queryByCommittee;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling.EntropyBasedActiveLearning;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * <p>
 * Title: VoteEntropyQueryByCommittee
 * </p>
 * 
 * <p>
 * Description: queries controversial instances/regions in the input space.
 * Compares the label distributions of every candidate for a given set of
 * models. The winning candidate poses those label distributions where the
 * committee disagrees most.
 * 
 * Measure: Vote Entropy
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016 Jürgen Bernard
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.03
 */
public class VoteEntropyQueryByCommittee<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractQueryByCommitteeActiveLearning<O, FV> {

	public VoteEntropyQueryByCommittee(List<IClassifier<O, FV>> learningModels) {
		super(learningModels);
	}

	@Override
	public String getComparisonMethod() {
		return "Uses entropy to identify instances where models disagree";
	}

	@Override
	protected void calculateRanking(int count) {
		for (IClassifier<O, FV> classifier : learningModels)
			classifier.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		queryApplicabilities = new HashMap<>();
		remainingUncertainty = 0.0;

		// calculate overall score
		for (FV fv : learningCandidateFeatureVectors) {
			List<Map<String, Double>> labelDistributions = new ArrayList<>();
			for (IClassifier<O, FV> classifier : learningModels)
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

			if (distributions != null && distributions.size() > 0) {
				Map<String, Double> winningLabels = new HashMap();
				for (IClassifier<O, FV> classifier : learningModels) {
					List<String> test = classifier.test(Arrays.asList(fv));
					if (test != null && test.size() > 0) {
						String label = classifier.test(Arrays.asList(fv)).get(0);
						if (!winningLabels.containsKey(label))
							winningLabels.put(label, 1.0);
						else
							winningLabels.put(label, winningLabels.get(label) + 1.0);
					}
				}

				for (String label : winningLabels.keySet())
					winningLabels.put(label, winningLabels.get(label) / (double) learningModels.size());

				dist = EntropyBasedActiveLearning.calculateEntropy(winningLabels);
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
		System.out.println("VoteEntropyQueryByCommittee: remaining uncertainty = " + remainingUncertainty);
	}

	@Override
	public String getName() {
		return "Vote Entropy QBC";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}
