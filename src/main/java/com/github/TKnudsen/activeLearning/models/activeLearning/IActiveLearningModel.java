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
 * Copyright: (c) 2016-2017 Jürgen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.03
 */
public interface IActiveLearningModel<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>, Y> {

	@Deprecated
	/**
	 * There is doubt to leave it out. Training data is supposed to be out of
	 * scope for an active learner.
	 * 
	 * @param featureVectors
	 */
	public void setTrainingData(List<X> featureVectors);

	public void setLearningCandidates(List<X> featureVectors);

	@Deprecated
	/**
	 * There is doubt to leave it out. Training data is supposed to be out of
	 * scope for an active learner. Why should this be orchestrated in an active
	 * learner? Seems to yield overhead.
	 * 
	 * @param featureVectors
	 */
	public void addCandidateVectorToTrainingVector(X fv);

	public X suggestCandidate();

	public List<X> suggestCandidates(int count);

	public ILearningModel<O, X, Y> getLearningModel();

	public double getRemainingUncertainty();

	public double getCandidateApplicabilityScore(X featureVector);
}
