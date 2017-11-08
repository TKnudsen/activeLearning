package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.ComplexDataObject.model.tools.DataConversion;
import com.github.TKnudsen.DMandML.data.classification.IProbabilisticClassificationResultSupplier;
import com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

/**
 * <p>
 * Title: SimpsonsDiversityActiveLearningModel
 * </p>
 * 
 * <p>
 * Description:
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016-2017 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.03
 */
public class SimpsonsDiversityActiveLearningModel<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {
	protected SimpsonsDiversityActiveLearningModel() {
	}

	@Deprecated
	public SimpsonsDiversityActiveLearningModel(Classifier<O, FV> learningModel) {
		super(learningModel);
	}

	public SimpsonsDiversityActiveLearningModel(IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier) {
		super(classificationResultSupplier);
	}

	@Override
	protected void calculateRanking(int count) {
		IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier = getClassificationResultSupplier();
		if (classificationResultSupplier == null)
			learningModel.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		queryApplicabilities = new HashMap<>();
		remainingUncertainty = 0.0;

		// calculate overall score
		for (FV fv : learningCandidateFeatureVectors) {
			double v1 = getLabelProbabilityDiversity(fv);

			ranking.add(new EntryWithComparableKey<Double, FV>(v1, fv));

			queryApplicabilities.put(fv, 1 - v1);
			remainingUncertainty += (1 - v1);

			if (ranking.size() > count)
				ranking.remove(ranking.size() - 1);
		}

		remainingUncertainty /= (double) learningCandidateFeatureVectors.size();
		System.out.println("SimpsonsDiveristyActiveLearningModel: remaining uncertainty = " + remainingUncertainty);
	}

	/**
	 * anticipates the Simpson's Diversity index. Challenge: a probability
	 * distribution has to be abstracted to an array of integer-like values, all
	 * >=1.
	 * 
	 * @param labelDistribution
	 * @return
	 */
	public double getLabelProbabilityDiversity(FV fv) {
		IProbabilisticClassificationResultSupplier<FV> classificationResultSupplier = getClassificationResultSupplier();

		Map<String, Double> labelDistribution = null;
		if (classificationResultSupplier == null)
			labelDistribution = learningModel.getLabelDistribution(fv);
		else
			labelDistribution = classificationResultSupplier.get().getLabelDistribution(fv).getValueDistribution();

		if (labelDistribution == null)
			return 0;

		double[] histogram = DataConversion.toPrimitives(new ArrayList<>(labelDistribution.values()));

		if (histogram == null)
			return Double.NaN;

		double min = Double.MAX_VALUE;
		for (int i = 0; i < histogram.length; i++)
			if (histogram[i] > 0)
				min = Math.min(min, histogram[i]);

		if (min != Double.MAX_VALUE)
			min = 1 / min;
		else
			min = 1;

		double[] values = new double[histogram.length];
		for (int i = 0; i < histogram.length; i++)
			values[i] = histogram[i] * min;

		for (int i = 0; i < histogram.length; i++)
			if (values[i] > 0 && values[i] < 1.0)
				values[i] = Math.ceil(values[i]);

		double numberAll = 0;
		double simpsonIndex = 0;
		for (int i = 0; i < histogram.length; i++)
			if (histogram[i] > 0) {
				numberAll += histogram[i];
				simpsonIndex += histogram[i] * (histogram[i] - 1);
			}

		if (numberAll == 0)
			simpsonIndex = 0;
		else if (numberAll * (numberAll - 1) != 0)
			simpsonIndex /= numberAll * (numberAll - 1);
		else
			simpsonIndex = 1;
		return simpsonIndex;
	}

	@Override
	public String getName() {
		return "Simpsons Diversity";
	}

	@Override
	public String getDescription() {
		return getName();
	}
}
