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

import main.java.com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import main.java.com.github.TKnudsen.DMandML.model.supervised.classifier.ClassifierTools;
import main.java.com.github.TKnudsen.DMandML.model.supervised.classifier.WekaClassifierWrapper;

/**
 * @author Christian Ritter
 */
public class ExpectedLogLossReduction<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	public ExpectedLogLossReduction(Classifier<O, FV> learningModel) {
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
			if (map == null)
				continue;
			labels.addAll(map.keySet());
		}

		for (int i = 0; i < U; i++) {
			FV fv = learningCandidateFeatureVectors.get(i);
			Map<String, Double> dist = dists.get(i);

			double loss = 0;
			if (dist != null)
				for (String label : labels) {
					List<FV> newTrainingSet = new ArrayList<>();
					for (FV fv1 : learningCandidateFeatureVectors) {
						newTrainingSet.add((FV) fv1.clone());
					}
					fv = (FV) fv.clone();
					fv.add("class", label);
					newTrainingSet.add(fv);
					Classifier<O, FV> newClassifier = null;
					try {
						if (learningModel instanceof WekaClassifierWrapper) {
							newClassifier = ClassifierTools.createParameterizedCopy((WekaClassifierWrapper<O, FV>) learningModel);
							newClassifier.train(newTrainingSet, "class");
						} else
							throw new InstantiationException();
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
							Map<String, Double> d = newClassifier.getLabelDistribution(learningCandidateFeatureVectors.get(j));
							for (String l : d.keySet()) {
								sum += d.get(l) * Math.log(d.get(l));
							}
						}
					}
					sum *= -1;
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
		return "ExpectedLogLossReduction";
	}

	@Override
	public String getName() {
		return "ExpectedLogLossReduction";
	}
}