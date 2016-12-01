package com.github.TKnudsen.activeLearning.models.activeLearning.densityWeighting;

import com.github.TKnudsen.ComplexDataObject.data.entry.EntryWithComparableKey;
import com.github.TKnudsen.ComplexDataObject.data.features.numericalData.NumericalFeatureVector;
import com.github.TKnudsen.ComplexDataObject.data.ranking.Ranking;
import com.github.TKnudsen.activeLearning.models.activeLearning.AbstractActiveLearningModel;
import com.github.TKnudsen.activeLearning.models.learning.classification.IClassifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * For more information see: An Analysis of Active Learning Strategies for Sequence Labeling Tasks
 * by Burr Settles and Mark Craven section 3.4
 *
 * @author Christian Ritter
 */
public class InformationDensityActiveLearning extends AbstractActiveLearningModel {

    private AbstractActiveLearningModel baseModel;
    //keeping the density map can save time later
    private Map<NumericalFeatureVector, Double> density;
    private double beta = 1.0;

    public InformationDensityActiveLearning(IClassifier<Double, NumericalFeatureVector> learningModel,
                                            AbstractActiveLearningModel baseModel) {
        super(learningModel);
        this.baseModel = baseModel;
    }

    public InformationDensityActiveLearning(IClassifier<Double, NumericalFeatureVector> learningModel, double beta,
                                            AbstractActiveLearningModel baseModel) {
        super(learningModel);
        this.beta = beta;
        this.baseModel = baseModel;
    }

    public void setBaseModel(AbstractActiveLearningModel baseModel) {
        this.baseModel = baseModel;
    }

    public void setLearningCandidates(List<NumericalFeatureVector> featureVectors) {
        super.setLearningCandidates(featureVectors);
        density = null;
    }

    @Override
    protected void calculateRanking(int count) {
        ranking = new Ranking<>();
        remainingUncertainty = 0.0;

        if (learningCandidateFeatureVectors.size() < 1) return;

        int U = learningCandidateFeatureVectors.size();
        if (density == null) {
            density = new HashMap<>();
            for (int i = 0; i < U; i++) {
                double sim = 0.0;
                for (int j = 0; j < U; j++) {
                    if (i != j) {
                        sim += cosineSimilarity(learningCandidateFeatureVectors.get(i),
                                learningCandidateFeatureVectors.get(j));
                    }
                }
                sim /= U;
                density.put(learningCandidateFeatureVectors.get(i), sim);
            }
        }
        baseModel.setTrainingData(this.trainingFeatureVectors);
        baseModel.setLearningCandidates(this.learningCandidateFeatureVectors);
        baseModel.suggestCandidates(U);

        //the ranking has to contain EntryWithComparableKey where the keys are th2e informativeness of the feature vector
        Ranking<EntryWithComparableKey<Double, NumericalFeatureVector>> r = baseModel.getRanking();
        for (int i = 0; i < U; i++) {
            NumericalFeatureVector fv = r.get(i).getValue();
            double w = density.get(fv);
            double p = r.get(i).getKey();
            double uninterestingness = 1 - (1 - p) * w; //as high interest needs to have low values in Ranking
            ranking.add(new EntryWithComparableKey<>(uninterestingness, fv));
            remainingUncertainty += 1 - uninterestingness;
        }

        remainingUncertainty /= U;
        System.out.println("InformationDensityActiveLearning: remaining uncertainty = " + remainingUncertainty);
    }

    private double cosineSimilarity(NumericalFeatureVector fv1, NumericalFeatureVector fv2) {
        if (fv1.sizeOfFeatures() != fv2.sizeOfFeatures()) return 0.0;
        double[] v1 = fv1.getVector();
        double[] v2 = fv2.getVector();
        double a = 0, b = 0, c = 0;
        for (int i = 0; i < v1.length; i++) {
            a += v1[i] * v2[i];
            b += v1[i] * v1[i];
            c += v2[i] * v2[i];
        }
        return a / b / c;
    }
}
