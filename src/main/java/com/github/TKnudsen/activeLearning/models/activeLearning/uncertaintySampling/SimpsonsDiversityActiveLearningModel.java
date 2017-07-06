package com.github.TKnudsen.activeLearning.models.activeLearning.uncertaintySampling;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.ComplexDataObject.model.tools.DataConversion;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;

import main.java.com.github.TKnudsen.DMandML.model.supervised.classifier.Classifier;

public class SimpsonsDiversityActiveLearningModel<O, FV extends AbstractFeatureVector<O, ? extends Feature<O>>> extends AbstractActiveLearningModel<O, FV> {

	public SimpsonsDiversityActiveLearningModel(Classifier<O, FV> learningModel) {
		super(learningModel);
	}

	@Override
	protected void calculateRanking(int count) {
		learningModel.test(learningCandidateFeatureVectors);

		ranking = new Ranking<>();
		queryApplicabilities = new HashMap<>();
		remainingUncertainty = 0.0;

		// calculate overall score
		for (FV fv : learningCandidateFeatureVectors) {
			double v1 = getLabelProbabilityDiversity(learningModel.getLabelDistribution(fv));

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
	public double getLabelProbabilityDiversity(Map<String, Double> labelDistribution) {
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
