package com.github.TKnudsen.activeLearning.models.activeLearning.expectedErrorReduction;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

/**
 * @author Christian Ritter
 */
public class Expected01LossReduction<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	public Expected01LossReduction(IClassifier<O, FV> learningModel) {
		super(learningModel);
	}

	@Override
	protected void calculateRanking(int count) {
		ranking = new Ranking<>();
		remainingUncertainty = 0.0;

		if (learningCandidateFeatureVectors.size() < 1)
			return;

		int U = learningCandidateFeatureVectors.size();
		List<Map<String, Double>> dists = new ArrayList<>();
		for (FV fv : learningCandidateFeatureVectors) {
			dists.add(learningModel.getLabelDistribution(fv));
		}
		Set<String> labels = new HashSet<>();
		for (Map<String, Double> map : dists) {
			labels.addAll(map.keySet());
		}
		for (int i = 0; i < U; i++) {
			FV fv = learningCandidateFeatureVectors.get(i);
			Map<String, Double> dist = dists.get(i);
			double loss = 0;
			for (String label : labels) {
				List<FV> newTrainingSet = new ArrayList<>();
				for (FV fv1 : learningCandidateFeatureVectors) {
					newTrainingSet.add((FV) fv1.getCopy());
				}
				fv = (FV) fv.getCopy();
				fv.add("class", Double.valueOf(label));
				newTrainingSet.add(fv);
				IClassifier<O, FV> newClassifier = null;
				try {
					// newClassifier = learningModel.getClass().newInstance();
					newClassifier = learningModel.createParameterizedCopy();
					newClassifier.train(newTrainingSet, "class");
				} catch (InstantiationException | IllegalAccessException e) {
					e.printStackTrace();
				} catch (Exception e) {
					e.printStackTrace();
				}
				double sum = 0;
				for (int j = 0; j < U; j++) {
					if (newClassifier == null)
						break;
					if (i != j) {
						sum += 1 - newClassifier.getLabelProbabilityMax(learningCandidateFeatureVectors.get(j));
					}
				}
				loss += dist.get(label) * sum;
			}
			ranking.add(new EntryWithComparableKey<>(loss, fv));
			remainingUncertainty += loss;
		}
		remainingUncertainty /= U;
		System.out.println("Expected01LossReduction: remaining uncertainty = " + remainingUncertainty);
	}

	@Override
	public String getDescription() {
		return "Expected01LossReduction";
	}

	@Override
	public String getName() {
		return "Expected01LossReduction";
	}
}