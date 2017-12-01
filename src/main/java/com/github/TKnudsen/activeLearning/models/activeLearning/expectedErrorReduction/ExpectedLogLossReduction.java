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
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResult;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.data.classification.LabelDistribution;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.ClassifierTools;
import com.github.TKnudsen.DMandML.model.supervised.classifier.WekaClassifierWrapper;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;
import com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling.EntropyBasedActiveLearning;

/**
 * <p>
 * Title: ExpectedLogLossReduction
 * </p>
 * 
 * <p>
 * Description: Ranks potential learning candidates by estimating the expected
 * error reduction when labeling a candidate with its respective label
 * distribution. This is an implementation of the method proposed in Section 4.1
 * (Equation (4.2)) in "Active Learning", by Burr Settles (2012).
 * </p>
 * 
 * @author Christian Ritter
 * @version 1.02
 */
public class ExpectedLogLossReduction<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	private WekaClassifierWrapper<O, FV> parameterizedClassifier = null;

	@Deprecated
	public ExpectedLogLossReduction(Classifier<O, FV> learningModel) {
		super(learningModel);
	}

	/**
	 * Basic constructor. This active learning algorithm requires an instance of
	 * the classifier used for training (either the original or a new instance
	 * with identical parameterization). This classifier is NOT changed during
	 * active learning (it is only used to create a copy). Note: This is
	 * currently only implemented for {@link WekaClassifierWrapper}.
	 * 
	 * @param classificationResultSupplier
	 * @param parameterizedClassifier
	 */
	public ExpectedLogLossReduction(IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier, WekaClassifierWrapper<O, FV> parameterizedClassifier) {
		super(classificationResultSupplier);
		this.parameterizedClassifier = parameterizedClassifier;
	}

	@Override
	protected void calculateRanking(int count) {
		if (parameterizedClassifier == null || getClassificationResultSupplier() == null)
			calculateRankingOld(count);
		else {
			ranking = new Ranking<>();
			remainingUncertainty = 0.0;

			if (learningCandidateFeatureVectors.size() < 1)
				return;

			int U = learningCandidateFeatureVectors.size();

			List<LabelDistribution> dists = new ArrayList<>();
			for (FV fv : learningCandidateFeatureVectors) {
				dists.add(getClassificationResultSupplier().get().getLabelDistribution(fv));
			}

			Set<String> labels = new HashSet<>();
			for (LabelDistribution ld : dists) {
				if (ld == null)
					continue;
				labels.addAll(ld.getLabelSet());
			}

			for (int i = 0; i < U; i++) {
				FV fv = learningCandidateFeatureVectors.get(i);
				LabelDistribution dist = dists.get(i);

				double expectedError = 0.0;
				if (dist != null)
					for (String label : labels) {
						List<FV> newTrainingSet = new ArrayList<>();
						for (FV fv1 : learningCandidateFeatureVectors) {
							newTrainingSet.add(fv1);
						}
						FV fv2 = (FV) fv.clone();
						fv2.add(parameterizedClassifier.getClassAttribute(), label);
						newTrainingSet.add(fv2);
						Classifier<O, FV> newClassifier = null;
						try {
							newClassifier = ClassifierTools.createParameterizedCopy(parameterizedClassifier);
						} catch (Exception e) {
							e.printStackTrace();
						}
						newClassifier.train(newTrainingSet, parameterizedClassifier.getClassAttribute());
						expectedError += dist.getValueDistribution().get(label) * calculatelogloss(newClassifier.createClassificationResult(learningCandidateFeatureVectors));
					}
				ranking.add(new EntryWithComparableKey<>(expectedError, fv));
				if (ranking.size() > count) {
					ranking.removeLast();
				}
				remainingUncertainty += expectedError;
			}
			remainingUncertainty /= U;
			System.out.println("ExpectedLogLossReduction: remaining uncertainty = " + remainingUncertainty);
		}
	}

	private Double calculatelogloss(IProbabilisticClassificationResult<FV> classificationResult) {
		double loss = 0.0;
		for (FV fv : learningCandidateFeatureVectors) {
			loss += EntropyBasedActiveLearning.calculateEntropy(classificationResult.getLabelDistribution(fv).getValueDistribution());
		}
		return loss;
	}

	@Deprecated
	protected void calculateRankingOld(int count) {
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
					fv.add(learningModel.getClassAttribute(), label);
					newTrainingSet.add(fv);
					Classifier<O, FV> newClassifier = null;
					try {
						if (learningModel instanceof WekaClassifierWrapper) {
							newClassifier = ClassifierTools.createParameterizedCopy((WekaClassifierWrapper<O, FV>) learningModel);
							newClassifier.train(newTrainingSet, learningModel.getClassAttribute());
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