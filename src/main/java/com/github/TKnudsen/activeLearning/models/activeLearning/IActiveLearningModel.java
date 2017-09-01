package com.github.TKnudsen.activeLearning.models.activeLearning;

import java.util.List;

import com.github.TKnudsen.ComplexDataObject.data.features.AbstractFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.features.Feature;
import com.github.TKnudsen.DMandML.model.supervised.ILearningModel;

/**
 * <p>
 * Title: IActiveLearningModel
 * </p>
 * 
 * <p>
 * Description: algorithmic model that estimates the coverage of the feature
 * space with respect to already labeled features.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016-2017 J�rgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.03
 */
public interface IActiveLearningModel<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>, Y> {

	public void setTrainingData(List<X> featureVectors);

	public void setLearningCandidates(List<X> featureVectors);

	public void addCandidateVectorToTrainingVector(X fv);

	public X suggestCandidate();

	public List<X> suggestCandidates(int count);

	public ILearningModel<O, X, Y> getLearningModel();

	public double getRemainingUncertainty();

	public double getCandidateApplicabilityScore(X featureVector);
}
