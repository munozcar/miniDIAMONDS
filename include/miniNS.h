#ifndef CLUSTERER_H
#define CLUSTERER_H

#include <random>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <numeric>
#include <memory>
#include <functional>
#include <random>
#include <algorithm>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#include <unordered_set>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <array>

#define SWAPDOUBLE(a,b) {double copy; copy = a, a = b, b = copy;}
#define SWAPINT(a,b) {int copy; copy = a, a = b, b = copy;}
// -------------------------------------------------------------------------------------------------
using namespace std;
using namespace Eigen;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;

typedef Eigen::Ref<Eigen::ArrayXd> RefArrayXd;
typedef Eigen::Ref<Eigen::ArrayXXd> RefArrayXXd;
typedef Eigen::Ref<Eigen::ArrayXi> RefArrayXi;

// -------------------------------------------------------------------------------------------------
class Metric
{
    public:
        Metric(){};
        ~Metric(){};
        virtual double distance(RefArrayXd point1, RefArrayXd point2) = 0;

    protected:

    private:
};
// -------------------------------------------------------------------------------------------------
class Clusterer
{
    public:
        Clusterer(Metric &metric);
        ~Clusterer(){};

        virtual int cluster(RefArrayXXd sample, vector<int> &optimalClusterIndices, vector<int> &optimalClusterSizes) = 0;

    protected:
        Metric &metric;

    private:

};
// -------------------------------------------------------------------------------------------------
class KmeansClusterer : public Clusterer
{
    public:
        KmeansClusterer(Metric &metric, unsigned int minNclusters, unsigned int maxNclusters, unsigned int Ntrials, double relTolerance);
        ~KmeansClusterer();

        virtual int cluster(RefArrayXXd sample, vector<int> &optimalClusterIndices, vector<int> &optimalClusterSizes);


    protected:

    private:
        void chooseInitialClusterCenters(RefArrayXXd sample, RefArrayXXd centers, unsigned int Nclusters);
        bool updateClusterCentersUntilConverged(RefArrayXXd sample, RefArrayXXd centers,
                                                RefArrayXd clusterSizes, vector<int> &clusterIndices,
                                                double &sumOfDistancesToClosestCenter, double relTolerance);
        double evaluateBICvalue(RefArrayXXd sample, RefArrayXXd centers, RefArrayXd clusterSizes,
                                vector<int> &clusterIndices);

        unsigned int minNclusters;
        unsigned int maxNclusters;
        unsigned int Ntrials;
        double relTolerance;
        mt19937 engine;
};
// -------------------------------------------------------------------------------------------------
class EuclideanMetric : public Metric
{
    public:
        EuclideanMetric(){};
        ~EuclideanMetric(){};
        virtual double distance(RefArrayXd point1, RefArrayXd point2);

    protected:

    private:
};
// -------------------------------------------------------------------------------------------------
namespace Functions
{
    const double PI = 4.0 * atan(1.0);

    // Profile functions
    void lorentzProfile(RefArrayXd predictions, const RefArrayXd covariates,
                        const double centroid = 0.0, const double amplitude = 1.0, const double gamma = 1.0);
    void modeProfile(RefArrayXd predictions, const RefArrayXd covariates,
                     const double centroid = 0.0, const double height = 1.0, const double linewidth = 1.0);
    void modeProfileWithAmplitude(RefArrayXd predictions, const RefArrayXd covariates,
                                  const double centroid = 0.0, const double amplitude = 1.0, const double linewidth = 1.0);
    void modeProfileSinc(RefArrayXd predictions, const RefArrayXd covariates,
                     const double centroid = 0.0, const double height = 1.0, const double resolution = 1.0);
    double logGaussProfile(const double covariate, const double mu = 0.0,
                           const double sigma = 1.0, const double amplitude = 1.0);
    void logGaussProfile(RefArrayXd predictions, const RefArrayXd covariates, const double mu = 0.0,
                         const double sigma = 1.0, const double amplitude = 1.0);

    // Likelihood functions

    double logGaussLikelihood(const RefArrayXd observations, const RefArrayXd predictions, const RefArrayXd uncertainties);

    // Matrix algebra functions
    void clusterCovariance(RefArrayXXd const clusterSample, RefArrayXXd covarianceMatrix,
                           RefArrayXd centerCoordinates);
    bool selfAdjointMatrixDecomposition(RefArrayXXd const covarianceMatrix, RefArrayXd eigenvalues,
                                        RefArrayXXd eigenvectorsMatrix);


    // Array manipulation functions
    inline double product(const vector<double> &vec);
    inline double sum(const vector<double> &vec);
    double logExpSum(const double x, const double y);
    double logExpDifference(const double x, const double y);
    void topDownMerge(RefArrayXd array1, RefArrayXd arrayCopy1,         // Only used within topDownMergeSort
                      RefArrayXd array2, RefArrayXd arrayCopy2,
                      int beginIndex, int middleIndex, int endIndex);
    void topDownSplitMerge(RefArrayXd array1, RefArrayXd arrayCopy1,    // Only used within topDownMergeSort
                           RefArrayXd array2, RefArrayXd arrayCopy2,
                           int beginIndex, int endIndex);
    void topDownMergeSort(RefArrayXd array1, RefArrayXd array2);        // Mergesort algorithm to sort array1 in ascending order and array2 according to array1
    void sortElementsDouble(RefArrayXd array1, RefArrayXd array2);
    void sortElementsInt(vector<int> &array1, RefArrayXd array2);


    vector<int> findArrayIndicesWithinBoundaries(RefArrayXd const array, double lowerBound, double upperBound);
    int countArrayIndicesWithinBoundaries(RefArrayXd const array, double lowerBound, double upperBound);
    ArrayXd cubicSplineInterpolation(RefArrayXd const observedAbscissa, RefArrayXd const observedOrdinate,
                                     RefArrayXd const interpolatedAbscissaUntruncated);


    // Utility functions

    template <typename Type>
    vector<int> argsort(const vector<Type> &myVector);
}

template <typename Type>
vector<int> Functions::argsort(const vector<Type> &myVector)
{
    vector<int> indices(myVector.size());
    iota(begin(indices), end(indices), 0);
    sort(begin(indices), end(indices), [&myVector] (int i, int j) {return myVector[i] < myVector[j];} );

    return indices;
}
// -------------------------------------------------------------------------------------------------
class Ellipsoid
{
    public:
        Ellipsoid(RefArrayXXd sampleOfParameters, const double enlargementFraction = 0.0); // Default no enlargement
        ~Ellipsoid();

        void resetEnlargementFraction(const double newEnlargementFraction);
        bool overlapsWith(Ellipsoid ellipsoid, bool &ellipsoidMatrixDecompositionIsSuccessful);
        bool containsPoint(const RefArrayXd pointCoordinates);
        void drawPoint(RefArrayXd drawnPoint);
        ArrayXd getCenterCoordinates();
        ArrayXd getEigenvalues();
        ArrayXXd getSample();
        ArrayXXd getCovarianceMatrix();
        ArrayXXd getEigenvectors();
        int getSampleSize();
        double getHyperVolume();
        double getEnlargementFraction();

    protected:
        ArrayXd centerCoordinates;
        ArrayXd originalEigenvalues;        // non-enlarged eigenvalues
        ArrayXd enlargedEigenvalues;        // enlarged eigenvalues
        ArrayXXd sample;
        ArrayXXd covarianceMatrix;
        ArrayXXd eigenvectors;
        int sampleSize;
        double hyperVolume;
        double enlargementFraction;

    private:
        int Ndimensions;
        mt19937 engine;
        uniform_real_distribution<> uniform;
        normal_distribution<> normal;
};
// -------------------------------------------------------------------------------------------------
class Model
{
    public:
        Model(const RefArrayXd covariates);
        ~Model();
        ArrayXd getCovariates();

        virtual void predict(RefArrayXd predictions, const RefArrayXd modelParameters) = 0;
        int getNparameters();

    protected:
        ArrayXd covariates;
        int Nparameters;

    private:
}; // END class Model
// -------------------------------------------------------------------------------------------------
class Likelihood
{
    public:
        Likelihood(const RefArrayXd observations, Model &model);
        ~Likelihood();
        ArrayXd getObservations();
        virtual double logValue(RefArrayXd const modelParameters) = 0;

    protected:
        ArrayXd observations;
        Model &model;

    private:
};
// -------------------------------------------------------------------------------------------------
class NormalLikelihood : public Likelihood
{

    public:
        NormalLikelihood(const RefArrayXd observations, const RefArrayXd uncertainties, Model &model);
        ~NormalLikelihood();
        ArrayXd getUncertainties();

        virtual double logValue(RefArrayXd const modelParameters);

    private:
        ArrayXd uncertainties;

};
// -------------------------------------------------------------------------------------------------
namespace File
{
    void openInputFile(ifstream &inputFile, string inputFileName);
    void openOutputFile(ofstream &outputFile, string outputFileName);

    ArrayXXd arrayXXdFromFile(ifstream &inputFile, const unsigned long Nrows, const int Ncols, char separator = ' ', char commentChar = '#');
    vector<string> vectorStringFromFile(ifstream &inputFile, const unsigned long Nrows, char commentChar = '#');

    void arrayXXdToFile(ofstream &outputFile, RefArrayXXd array, string separator = "  ", string terminator = "\n");
    void twoArrayXdToFile(ofstream &outputFile, RefArrayXd array1, RefArrayXd array2, string separator = "  ", string terminator = "\n");
    void arrayXdToFile(ofstream &outputFile, RefArrayXd array, string terminator = "\n");
    void arrayXXdRowsToFiles(RefArrayXXd array, string fullPathPrefix, string fileExtension = ".txt", string terminator = "\n");
    void sniffFile(ifstream &inputFile, unsigned long &Nrows, int &Ncols, char separator = ' ', char commentChar = '#');

}
// -------------------------------------------------------------------------------------------------
class Prior
{
    public:

        Prior(const int Ndimensions);
        ~Prior();
        int getNdimensions();

        virtual double logDensity(RefArrayXd const x, const bool includeConstantTerm = false) = 0;
        virtual bool drawnPointIsAccepted(RefArrayXd const drawnPoint) = 0;
        virtual void draw(RefArrayXXd drawnSample) = 0;
        virtual void drawWithConstraint(RefArrayXd drawnPoint, Likelihood &likelihood) = 0;
        virtual void writeHyperParametersToFile(string fullPath) = 0;

        const double minusInfinity;

    protected:
        int Ndimensions;
        mt19937 engine;

    private:
}; // END class Prior
// -------------------------------------------------------------------------------------------------
class UniformPrior : public Prior
{
    public:

        UniformPrior(const RefArrayXd minima, const RefArrayXd maxima);
        ~UniformPrior();

        ArrayXd getMinima();
        ArrayXd getMaxima();

        virtual double logDensity(RefArrayXd const x, const bool includeConstantTerm = false);
        virtual bool drawnPointIsAccepted(RefArrayXd const drawnPoint);
        virtual void draw(RefArrayXXd drawnSample);
        virtual void drawWithConstraint(RefArrayXd drawnPoint, Likelihood &likelihood);
        virtual void writeHyperParametersToFile(string fullPath);

    private:
        uniform_real_distribution<> uniform;
        ArrayXd minima;
        ArrayXd maxima;
}; // END class UniformPrior
// -------------------------------------------------------------------------------------------------
class NestedSampler ;
// -------------------------------------------------------------------------------------------------

class LivePointsReducer
{
    public:
        LivePointsReducer(NestedSampler &nestedSampler);
        ~LivePointsReducer();

        vector<int> findIndicesOfLivePointsToRemove(mt19937 engine);
        int getNlivePointsToRemove();

        virtual int updateNlivePoints() = 0;

    protected:
        int NlivePointsAtCurrentIteration;
        int updatedNlivePoints;
        NestedSampler &nestedSampler;

    private:
        int NlivePointsToRemove;

};// -------------------------------------------------------------------------------------------------
class NestedSampler
{
    public:

        NestedSampler(const bool printOnTheScreen, const int initialNlivePoints, const int minNlivePoints, vector<Prior*> ptrPriors,
                      Likelihood &likelihood, Metric &metric, Clusterer &clusterer);
        ~NestedSampler();

        void run(LivePointsReducer &livePointsReducer, const int NinitialIterationsWithoutClustering = 100,
                 const int NiterationsWithSameClustering = 50, const int maxNdrawAttempts = 5000,
                 const double maxRatioOfRemainderToCurrentEvidence = 0.05, string pathPrefix = "");

        virtual bool drawWithConstraint(const RefArrayXXd totalSample, const unsigned int Nclusters, const vector<int> &clusterIndices,
                                        const vector<int> &clusterSizes, RefArrayXd drawnPoint,
                                        double &logLikelihoodOfDrawnPoint, const int maxNdrawAttempts) = 0;


        // Define set and get functions

        unsigned int getNiterations();
        unsigned int getNdimensions();
        int getNlivePoints();
        int getInitialNlivePoints();
        int getMinNlivePoints();
        double getLogCumulatedPriorMass();
        double getLogRemainingPriorMass();
        double getRatioOfRemainderToCurrentEvidence();
        double getLogMaxLikelihoodOfLivePoints();
        double getComputationalTime();
        double getTerminationFactor();
        vector<int> getNlivePointsPerIteration();
        ArrayXXd getNestedSample();
        ArrayXd getLogLikelihood();

        void setLogEvidence(double newLogEvidence);
        double getLogEvidence();

        void setLogEvidenceError(double newLogEvidenceError);
        double getLogEvidenceError();

        void setInformationGain(double newInformationGain);
        double getInformationGain();

        void setPosteriorSample(ArrayXXd newPosteriorSample);
        ArrayXXd getPosteriorSample();

        void setLogLikelihoodOfPosteriorSample(ArrayXd newLogLikelihoodOfPosteriorSample);
        ArrayXd getLogLikelihoodOfPosteriorSample();

        void setLogWeightOfPosteriorSample(ArrayXd newLogWeightOfPosteriorSample);
        ArrayXd getLogWeightOfPosteriorSample();

        void setOutputPathPrefix(string newOutputPathPrefix);
        string getOutputPathPrefix();

        ofstream outputFile;                        // An output file stream to save configuring parameters also from derived classes


    protected:

        vector<Prior*> ptrPriors;                   // A vector of pointers to objects of class Prior, containing the priors for each parameter
        Likelihood &likelihood;                     // An object of class Likelihood to contain the likelihood used in the Bayesian inference
        Metric &metric;                             // An object of class Metric for the proper metric to adopt in the computation
        Clusterer &clusterer;                       // An object of class Clusterer to contain the cluster algorithm used in the process
        bool printOnTheScreen;                      // A boolean specifying whether we want current results to be printed on the screen
        unsigned int Ndimensions;                   // Total number of dimensions of the inference
        int NlivePoints;                            // Total number of live points at a given iteration
        int minNlivePoints;                         // Minimum number of live points allowed
        double worstLiveLogLikelihood;              // The worst likelihood value of the current live sample
        double logCumulatedPriorMass;               // The total (cumulated) prior mass at a given nested iteration
        double logRemainingPriorMass;               // The remaining width in prior mass at a given nested iteration (log X)
        double ratioOfRemainderToCurrentEvidence;   // The current ratio of live to cumulated evidence
        vector<int> NlivePointsPerIteration;           // A vector that stores the number of live points used at each iteration of the nesting process

        mt19937 engine;

        virtual bool verifySamplerStatus() = 0;


	private:

        string outputPathPrefix;                 // The path of the directory where all the results have to be saved
        unsigned int Niterations;                // Counter saving the number of nested loops used
        int updatedNlivePoints;                  // The updated number of live points to be used in the next iteration
        int initialNlivePoints;                  // The initial number of live points
        double informationGain;                  // Skilling's Information gain in moving from prior to posterior PDF
        double logEvidence;                      // Skilling's Evidence
        double logEvidenceError;                 // Skilling's error on Evidence (based on IG)
        double logMaxLikelihoodOfLivePoints;     // The maximum log(Likelihood) of the set of live points
        double logMeanLikelihoodOfLivePoints;    // The logarithm of the mean likelihood value of the current set of live points
        double computationalTime;                // Computational time of the process
        double terminationFactor;                // The final value of the stopping condition for the nested process
        ArrayXXd nestedSample;                   // Parameter values (for all the free parameters of the problem) of the current set of live points
        ArrayXd logLikelihood;                   // log-likelihood values of the current set of live points
                                                 // is removed from the sample.
        ArrayXXd posteriorSample;                // Parameter values (for all the free parameters of the problem) in the final posterior sampling
        ArrayXd logLikelihoodOfPosteriorSample;  // log(Likelihood) values corresponding to the posterior sample
        ArrayXd logWeightOfPosteriorSample;      // log(Weights) = log(Likelihood) + log(dX) corresponding to the posterior sample

        void removeLivePointsFromSample(const vector<int> &indicesOfLivePointsToRemove,
                                        vector<int> &clusterIndices, vector<int> &clusterSizes);
        void printComputationalTime(const double startTime);
};
// -------------------------------------------------------------------------------------------------
class MultiEllipsoidSampler : public NestedSampler
{
    public:
        MultiEllipsoidSampler(const bool printOnTheScreen, vector<Prior*> ptrPriors,
                              Likelihood &likelihood, Metric &metric, Clusterer &clusterer,
                              const int initialNlivePoints, const int minNlivePoints,
                              const double initialEnlargementFraction, const double shrinkingRate);
        ~MultiEllipsoidSampler();

        virtual bool drawWithConstraint(const RefArrayXXd totalSample, const unsigned int Nclusters, const vector<int> &clusterIndices,
                                        const vector<int> &clusterSizes, RefArrayXd drawnPoint,
                                        double &logLikelihoodOfDrawnPoint, const int maxNdrawAttempts) override;

        virtual bool verifySamplerStatus() override;

        vector<Ellipsoid> getEllipsoids();
        double getInitialEnlargementFraction();
        double getShrinkingRate();


    protected:
        bool ellipsoidMatrixDecompositionIsSuccessful;  // A boolean specifying whether an error occurred in the
                                                        // eigenvalues decomposition of the ellipsoid matrix

        void computeEllipsoids(RefArrayXXd const totalSample, const unsigned int Nclusters,
                               const vector<int> &clusterIndices, const vector<int> &clusterSizes);
        void findOverlappingEllipsoids(vector<unordered_set<int>> &overlappingEllipsoidsIndices);
        double updateEnlargementFraction(const int clusterSize);

    private:
        vector<Ellipsoid> ellipsoids;
        int Nellipsoids;                        // Total number of ellipsoids computed
        double initialEnlargementFraction;      // Initial fraction for enlargement of ellipsoids
        double shrinkingRate;                   // Prior volume shrinkage rate (between 0 and 1)

        uniform_real_distribution<> uniform;
};
// -------------------------------------------------------------------------------------------------
class PowerlawReducer : public LivePointsReducer
{
    public:
        PowerlawReducer(NestedSampler &nestedSampler, const double tolerance = 100.0,
                           const double exponent = 1, const double terminationFactor = 0.05);
        ~PowerlawReducer();

        virtual int updateNlivePoints();

    protected:
        double tolerance;
        double exponent;
        double terminationFactor;

    private:
};
// -------------------------------------------------------------------------------------------------
class FerozReducer : public LivePointsReducer
{
    public:
        FerozReducer(NestedSampler &nestedSampler, const double tolerance);
        ~FerozReducer();
        virtual int updateNlivePoints();

    protected:
        double logMaxEvidenceContribution;          // The logarithm of the maximum evidence contribution at the previous iteration of the nesting process
        double tolerance;

    private:
};
// -------------------------------------------------------------------------------------------------
class Results
{
    public:
        Results(NestedSampler &nestedSampler);
        ~Results();

        void writeParametersToFile(string fileName, string outputFileExtension = ".txt");
        void writeLogLikelihoodToFile(string fileName);
        void writeLogWeightsToFile(string fileName);
        void writeEvidenceInformationToFile(string fileName);
        void writePosteriorProbabilityToFile(string fileName);
        void writeParametersSummaryToFile(string fileName, const double credibleLevel = 68.27, const bool writeMarginalDistribution = true);
        void writeObjectsIdentificationToFile(){};          // TO DO

    protected:
        double marginalDistributionMode;
        ArrayXd parameterValues;
        ArrayXd marginalDistribution;
        ArrayXd parameterValuesRebinned;
        ArrayXd parameterValuesInterpolated;
        ArrayXd marginalDistributionRebinned;
        ArrayXd marginalDistributionInterpolated;

    private:
        NestedSampler &nestedSampler;

        ArrayXd posteriorProbability();
        void writeMarginalDistributionToFile(const int parameterNumber);
        ArrayXd computeCredibleLimits(const double credibleLevel, const double skewness, const int NinterpolationsPerBin = 10);
        ArrayXXd parameterEstimation(const double credibleLevel, const bool writeMarginalDistribution);
};
#endif
