// To compile: clang++ -o test_miniNS test_miniNS.cpp -L./build/ -I ./include/ -l./minins -stdlib=libc++ -std=c++11 -Wno-deprecated-register


#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include "miniNS.h"

// -------------- "Parabolic" FORWARD MODEL ----------------------------------------------------------
class ParabolicModel : public Model
{
    public:
        ParabolicModel(const RefArrayXd covariates);
        ~ParabolicModel();
        ArrayXd getCovariates();
        virtual void predict(RefArrayXd predictions, const RefArrayXd modelParameters);

    protected:
    private:
};

ParabolicModel::ParabolicModel(const RefArrayXd covariates)
: Model(covariates)
{
}
ParabolicModel::~ParabolicModel()
{
}

void ParabolicModel::predict(RefArrayXd predictions, RefArrayXd const modelParameters)
{
    double a = modelParameters(0);
    double b = modelParameters(1);
    double c = modelParameters(2);
    double d = modelParameters(3);
    predictions = a*exp(-b*covariates) + c*covariates*log(d*covariates);
}


// -------------- "PSG RETRIEVAL" FORWARD MODEL ---------------------------------------------------
std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}
class PSGModel : public Model
{
    public:
        PSGModel(const RefArrayXd covariates);
        ~PSGModel();
        ArrayXd getCovariates();
        virtual void predict(RefArrayXd predictions, const RefArrayXd modelParameters);

    protected:
    private:
};
void PSGModel::predict(RefArrayXd predictions, RefArrayXd const modelParameters)
{
    std::string out = "";
    double planet_radius = modelParameters(0);
    double planet_temperature = modelParameters(1);
    exec("curl -d type=rad -d whdr=n --data-urlencode file@config.txt https://psg.gsfc.nasa.gov/api.php");
}

// -------------- MAIN PROGRAM --------------------------------------------------------------------

int main(int argc, char *argv[])
{

    // Check number of arguments for main function
    if (argc != 1)
    {
        cerr << "Usage: ./miniNS" << endl;
        exit(EXIT_FAILURE);
    }
    // Read input data
    unsigned long Nrows;
    int Ncols;
    ArrayXXd data;
    string baseInputDirName = "";
    string inputFileName = "exponential.txt"; // data
    string outputPathPrefix = "Exponential_";

    ifstream inputFile;
    File::openInputFile(inputFile, inputFileName);
    File::sniffFile(inputFile, Nrows, Ncols);
    data = File::arrayXXdFromFile(inputFile, Nrows, Ncols);
    inputFile.close();

    // Create arrays for each data type
    ArrayXd covariates = data.col(0);
    ArrayXd observations = data.col(1);
    ArrayXd uncertainties = data.col(2);

    // Uniform Prior
    unsigned long Nparameters;  // Number of parameters for which prior distributions are defined

    int Ndimensions = 4;        // Number of free parameters (dimensions) of the problem
    vector<Prior*> ptrPriors(1);
    ArrayXd parametersMinima(Ndimensions);
    ArrayXd parametersMaxima(Ndimensions);
    parametersMinima <<  5, 0, 0, 0;         // Minima values for the free parameters
    parametersMaxima << 15, 8, 6, 1;     // Maxima values for the free parameters

    UniformPrior uniformPrior(parametersMinima, parametersMaxima);
    ptrPriors[0] = &uniformPrior;

    string fullPathHyperParameters = outputPathPrefix + "hyperParametersUniform.txt";       // Print prior hyper parameters as output
    uniformPrior.writeHyperParametersToFile(fullPathHyperParameters);

    // Set up the models for the inference problem
    ParabolicModel model(covariates);      // Parabolic function


    // Set up the likelihood function to be used
    NormalLikelihood likelihood(observations, uncertainties, model);

    // Set up the K-means clusterer using an Euclidean metric

    int minNclusters = 1;  // Minimum number of clusters
    int maxNclusters = 10; // Maximum number of clusters

    int Ntrials = 10;
    double relTolerance = 0.01;

    EuclideanMetric myMetric;
    KmeansClusterer kmeans(myMetric, minNclusters, maxNclusters, Ntrials, relTolerance);

    // Configure and start nested sampling inference -----

    bool printOnTheScreen = true;                  // Print results on the screen
    int initialNobjects = 1000;                     // Initial number of live points
    int minNobjects = 1000;                         // Minimum number of live points
    int maxNdrawAttempts = 10000;                  // Maximum number of attempts when trying to draw a new sampling point
    int NinitialIterationsWithoutClustering = 500; // The first N iterations, we assume that there is only 1 cluster
    int NiterationsWithSameClustering = 50;        // Clustering is only happening every N iterations.
    double initialEnlargementFraction = 0.45;      // Fraction by which each axis in an ellipsoid has to be enlarged.
                                                            // It can be a number >= 0, where 0 means no enlargement.
    double shrinkingRate = 0.02;                   // Exponent for remaining prior mass in ellipsoid enlargement fraction.
                                                            // It is a number between 0 and 1. The smaller the slower the shrinkage
                                                            // of the ellipsoids.

    double terminationFactor = 0.01;    // Termination factor for nested sampling process.

    // RUN NESTED SAMPLER
    MultiEllipsoidSampler nestedSampler(printOnTheScreen, ptrPriors, likelihood, myMetric, kmeans,
                                        initialNobjects, minNobjects, initialEnlargementFraction, shrinkingRate);

    double tolerance = 1.e2;
    double exponent = 0.4;
    PowerlawReducer livePointsReducer(nestedSampler, tolerance, exponent, terminationFactor);

    nestedSampler.run(livePointsReducer, NinitialIterationsWithoutClustering, NiterationsWithSameClustering,
                      maxNdrawAttempts, terminationFactor, outputPathPrefix);

    nestedSampler.outputFile << "# List of configuring parameters used for the ellipsoidal sampler and X-means" << endl;
    nestedSampler.outputFile << "# Row #1: Minimum Nclusters" << endl;
    nestedSampler.outputFile << "# Row #2: Maximum Nclusters" << endl;
    nestedSampler.outputFile << "# Row #3: Initial Enlargement Fraction" << endl;
    nestedSampler.outputFile << "# Row #4: Shrinking Rate" << endl;
    nestedSampler.outputFile << minNclusters << endl;
    nestedSampler.outputFile << maxNclusters << endl;
    nestedSampler.outputFile << initialEnlargementFraction << endl;
    nestedSampler.outputFile << shrinkingRate << endl;
    nestedSampler.outputFile.close();

    // Save the results in output files -----

    Results results(nestedSampler);
    results.writeParametersToFile("parameter");
    results.writeLogLikelihoodToFile("logLikelihood.txt");
    results.writeLogWeightsToFile("logWeights.txt");
    results.writeEvidenceInformationToFile("evidenceInformation.txt");
    results.writePosteriorProbabilityToFile("posteriorDistribution.txt");

    double credibleLevel = 68.3;
    bool writeMarginalDistributionToFile = true;
    results.writeParametersSummaryToFile("parameterSummary.txt", credibleLevel, writeMarginalDistributionToFile);

    cout << "miniNS done." << endl;

    return EXIT_SUCCESS;
}
