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
 * Copyright: (c) 2016 Jürgen Bernard https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.03
 */
public class VoteEntropyQueryByCommittee extends AbstractQueryByCommitteeActiveLearning {

	public VoteEntropyQueryByCommittee(List<IClassifier<Double, NumericalFeatureVector>> learningModels) {
		super(learningModels);
	}

	@Override
	public String getComparisonMethod() {
		return "Uses entropy to identify instances where models disagree";
	}

	@Override
	protected void calculateRanking(int count) {
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

			// update ranking
			ranking.add(new EntryWithComparableKey<Double, NumericalFeatureVector>(1 - dist, fv));
			remainingUncertainty += dist;

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("VoteEntropyQueryByCommittee: remaining uncertainty = " + remainingUncertainty);
	}
}
