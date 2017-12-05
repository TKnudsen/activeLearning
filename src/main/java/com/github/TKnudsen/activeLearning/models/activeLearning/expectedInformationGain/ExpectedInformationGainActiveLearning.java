package com.github.TKnudsen.activeLearning.models.activeLearning.expectedInformationGain;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.ComplexDataObject.model.statistics.Entropy;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.data.classification.LabelDistribution;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.ClassifierTools;
import com.github.TKnudsen.DMandML.model.supervised.classifier.WekaClassifierWrapper;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

/**
 * <p>
 * Title: ExpectedInformationGainActiveLearning
 * </p>
 * 
 * <p>
 * Description: Ranks potential learning candidates by estimating the expected
 * information gain when labeling a candidate with its respective label
 * distribution. This is an implementation of the method proposed in Section 6.1
 * (Equation (6.2)) in "Active Learning", by Burr Settles (2012).
 * </p>
 * 
 * @author Christian Ritter
 * @version 1.03
 */
public class ExpectedInformationGainActiveLearning<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	private Classifier<O, FV> parameterizedClassifier = null;
	private Supplier<List<FV>> trainingDataSupplier;

	/**
	 * Basic constructor. This active learning algorithm requires an instance of
	 * the classifier used for training (either the original or a new instance
	 * with identical parameterization). If, and only if, this classifier is
	 * extending {@link WekaClassifierWrapper} it is not changed during active
	 * learning (it then uses a parameterized copy).
	 * 
	 * @param classificationResultSupplier
	 * @param parameterizedClassifier
	 * @param trainingDataSupplier
	 */
	public ExpectedInformationGainActiveLearning(IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier, Classifier<O, FV> parameterizedClassifier, Supplier<List<FV>> trainingDataSupplier) {
		super(classificationResultSupplier);
		this.parameterizedClassifier = parameterizedClassifier;
		this.trainingDataSupplier = trainingDataSupplier;
	}

	@Override
	protected void calculateRanking(int count) {
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

		boolean moreThanOneLabel = trainingDataSupplier.get().stream().map(x -> x.getAttribute(parameterizedClassifier.getClassAttribute())).collect(Collectors.toSet()).size() > 1;

		for (int i = 0; i < U; i++) {
			FV fv = learningCandidateFeatureVectors.get(i);
			LabelDistribution dist = dists.get(i);

			double informationGain = Entropy.calculateEntropy(getClassificationResultSupplier().get().getLabelDistribution(fv).getValueDistribution());
			// only useful if more than one label is set
			if (moreThanOneLabel) {
				if (dist != null)
					for (String label : labels) {
						List<FV> newTrainingSet = new ArrayList<>();
						for (FV fv1 : trainingDataSupplier.get()) {
							newTrainingSet.add(fv1);
						}
						FV fv2 = (FV) fv.clone();
						fv2.add(parameterizedClassifier.getClassAttribute(), label);
						newTrainingSet.add(fv2);
						Classifier<O, FV> newClassifier = null;
						try {
							if (parameterizedClassifier instanceof WekaClassifierWrapper)
								newClassifier = ClassifierTools.createParameterizedCopy((WekaClassifierWrapper<O, FV>) parameterizedClassifier);
							else
								newClassifier = parameterizedClassifier;
						} catch (Exception e) {
							e.printStackTrace();
						}
						try {
							newClassifier.train(newTrainingSet, parameterizedClassifier.getClassAttribute());
							informationGain -= dist.getValueDistribution().get(label) * Entropy.calculateEntropy(newClassifier.createClassificationResult(learningCandidateFeatureVectors).getLabelDistribution(fv).getValueDistribution());
						} catch (Exception e) {
						}
					}
			}
			ranking.add(new EntryWithComparableKey<>(-informationGain, fv));
			if (ranking.size() > count) {
				ranking.removeLast();
			}
			remainingUncertainty -= informationGain;
		}
		remainingUncertainty /= U;
		System.out.println("ExpectedInformationGainActiveLearning: remaining uncertainty = " + remainingUncertainty);

	}

	@Override
	public String getDescription() {
		return "ExpectedInformationGainActiveLearning";
	}

	@Override
	public String getName() {
		return "ExpectedInformationGainActiveLearning";
	}
}