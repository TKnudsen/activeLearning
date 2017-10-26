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
 * Description: active learners suggest unlabeled instances to oracles (i.e.,
 * users) in a way that a given learning model (e.g., a classifier) is supposed
 * to improce its quality in a "best" way. Formalization of "best" depends on
 * the particular implementation.
 * </p>
 * 
 * <p>
 * Copyright: (c) 2016-2017 Juergen Bernard,
 * https://github.com/TKnudsen/activeLearning
 * </p>
 * 
 * @author Juergen Bernard
 * @version 1.05
 */
public interface IActiveLearningModel<O, X extends AbstractFeatureVector<O, ? extends Feature<O>>, Y> {

	public void setLearningCandidates(List<X> featureVectors);

	public X suggestCandidate();

	public List<X> suggestCandidates(int count);

	public ILearningModel<O, X, Y> getLearningModel();

	public double getRemainingUncertainty();

	public double getCandidateApplicabilityScore(X featureVector);
}
