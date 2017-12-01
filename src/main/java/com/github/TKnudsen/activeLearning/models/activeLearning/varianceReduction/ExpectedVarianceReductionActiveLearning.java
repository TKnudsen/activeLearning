package com.github.TKnudsen.activeLearning.models.activeLearning.varianceReduction;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.ComplexDataObject.model.tools.StatisticsSupport;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResult;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.data.classification.LabelDistribution;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.ClassifierTools;
import com.github.TKnudsen.DMandML.model.supervised.classifier.WekaClassifierWrapper;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

/**
 * <p>
 * Title: VarianceReductionActiveLearning
 * </p>
 * 
 * <p>
 * Description: Ranks potential learning candidates by estimating the reduction
 * in the variance of the resulting label distributions when labeling a
 * candidate with its respective label distribution. This is an implementation
 * of the method proposed in Section 4.2 (Equation (4.4)) in "Active Learning",
 * by Burr Settles (2012).
 * </p>
 * 
 * @author Christian Ritter
 * @version 1.01
 */
public class ExpectedVarianceReductionActiveLearning<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	private WekaClassifierWrapper<O, FV> parameterizedClassifier = null;

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
	public ExpectedVarianceReductionActiveLearning(IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier, WekaClassifierWrapper<O, FV> parameterizedClassifier) {
		super(classificationResultSupplier);
		this.parameterizedClassifier = parameterizedClassifier;
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

		for (int i = 0; i < U; i++) {
			FV fv = learningCandidateFeatureVectors.get(i);
			LabelDistribution dist = dists.get(i);

			double variance = 0.0;
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
					variance += dist.getValueDistribution().get(label) * calculateExpectedVariance(newClassifier.createClassificationResult(learningCandidateFeatureVectors));
				}
			ranking.add(new EntryWithComparableKey<>(variance, fv));
			if (ranking.size() > count) {
				ranking.removeLast();
			}
			remainingUncertainty += variance;
		}
		remainingUncertainty /= U;
		System.out.println("ExpectedVarianceReductionActiveLearning: remaining uncertainty = " + remainingUncertainty);

	}

	private Double calculateExpectedVariance(IProbabilisticClassificationResult<FV> classificationResult) {
		double variance = 0.0;
		for (FV fv : learningCandidateFeatureVectors) {
			StatisticsSupport stats = new StatisticsSupport(classificationResult.getLabelDistribution(fv).getValueDistribution().values());
			variance += stats.getVariance();
		}
		return variance;
	}

	@Override
	public String getDescription() {
		return "ExpectedVarianceReductionActiveLearning";
	}

	@Override
	public String getName() {
		return "ExpectedVarianceReductionActiveLearning";
	}
}