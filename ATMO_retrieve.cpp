// To compile: clang++ -o ATMO_retrieve ATMO_retrieve.cpp -L./build/ -I ./include/ -l minins -stdlib=libc++ -std=c++11 -Wno-deprecated-register


#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include "miniNS.h"

// -------------- "ATMO GRID" FORWARD MODEL ----------------------------------------------------------
class ATMOModel : public Model
{
    public:
        ATMOModel(const RefArrayXd covariates);
        ~ATMOModel();
        ArrayXd getCovariates();
        virtual void predict(RefArrayXd predictions, const RefArrayXd modelParameters);

    protected:
    private:
};

ATMOModel::ATMOModel(const RefArrayXd covariates)
: Model(covariates)
{
}
ATMOModel::~ATMOModel()
{
}

void ATMOModel::predict(RefArrayXd predictions, RefArrayXd const modelParameters)
{
    string model_name = "../ATMO/transfiles_txt/trans-iso-generic__10_+1.7__1100_1.00_model.txt";

    int temp_num = round(modelParameters(0)/100)*100; // Temperature
    float co_num = modelParameters(1); // C/O ratio
    if (co_num < 0.455)
    {
      co_num = 0.35;}
      else if (co_num < 0.63)
      {
        co_num = 0.56;}
        else if (co_num < 0.85)
        {
          co_num = 0.70;}
          else{
            co_num = 1.00;}

    string CO_Ratio = std::to_string(co_num);
    string Temperature = std::to_string(temp_num);

    int insert_pos_t = 41;
    int insert_pos_co = 54;
    model_name.insert (insert_pos_t, Temperature);
    model_name.insert (insert_pos_co, CO_Ratio, 0, 4);
    if (temp_num < 1000)
    {
      model_name.insert (insert_pos_t, "0");
    }

    unsigned long Nrows;
    int Ncols;

    ifstream modelFile;
    File::openInputFile(modelFile, model_name);
    File::sniffFile(modelFile, Nrows, Ncols);
    ArrayXXd model = File::arrayXXdFromFile(modelFile, Nrows, Ncols);
    modelFile.close();

    predictions = model.col(1).block(0,0,200,1);
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
    string inputFileName = "input_model.txt"; // data
    string outputPathPrefix = "Transmission_";

    ifstream inputFile;
    File::openInputFile(inputFile, inputFileName);
    File::sniffFile(inputFile, Nrows, Ncols);
    data = File::arrayXXdFromFile(inputFile, Nrows, Ncols);
    inputFile.close();

    // Create arrays for each data type
    ArrayXd covariates = data.col(0).block(0,0,200,1);
    ArrayXd observations = data.col(1).block(0,0,200,1);
    ArrayXd uncertainties = data.col(2).block(0,0,200,1);

    // Uniform Prior
    unsigned long Nparameters;  // Number of parameters for which prior distributions are defined

    int Ndimensions = 2;        // Number of free parameters (dimensions) of the problem
    vector<Prior*> ptrPriors(1);
    ArrayXd parametersMinima(Ndimensions);
    ArrayXd parametersMaxima(Ndimensions);
    parametersMinima <<  1000, 0;         // Minima values for the free parameters
    parametersMaxima << 2000, 1;     // Maxima values for the free parameters

    UniformPrior uniformPrior(parametersMinima, parametersMaxima);
    ptrPriors[0] = &uniformPrior;

    string fullPathHyperParameters = outputPathPrefix + "hyperParametersUniform.txt";       // Print prior hyper parameters as output
    uniformPrior.writeHyperParametersToFile(fullPathHyperParameters);

    // Set up the models for the inference problem
    ATMOModel model(covariates);      // ATMO function


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
