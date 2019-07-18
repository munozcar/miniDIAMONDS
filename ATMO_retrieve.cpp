
// To compile: clang++ -o ATMO_retrieve ATMO_retrieve.cpp -L./build/ -I ./include/ -l minins -stdlib=libc++ -std=c++11 -Wno-deprecated-register

#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

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

  int file_descriptor;
  size_t length;
  void *gridp;                                 // Grid will have values of type double.
  struct stat sb;

  // Open file with data.
  file_descriptor = open("./ATMO_GRID_NORMALIZED.txt", O_RDONLY);

  // Calculate size of grid & check for reading error.
  if (fstat(file_descriptor, &sb) == -1) {
    printf("MAP FAILED");
    exit(EXIT_FAILURE);
  }
  length = sb.st_size; // Length, in bytes.

  // Map file into RAM
  gridp = mmap(NULL, length, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
  double *grid = (double *)gridp;
  if (grid == MAP_FAILED) {
    printf("MAP FAILED");
    exit(EXIT_FAILURE);
  }

  double interpolation_params[6];
  double interpolated_model[5000];

  interpolation_params[0] = modelParameters[0];
  interpolation_params[1] = modelParameters[1];
  interpolation_params[2] = modelParameters[2];
  interpolation_params[3] = modelParameters[3];
  interpolation_params[4] = modelParameters[4];
  interpolation_params[5] = modelParameters[5];

  multipolator(grid, interpolation_params, interpolated_model);



  for (int i=0; i<4999; i++){
    predictions[i] = interpolated_model[i];
  }

  // Remove RAM mapping, close file. Fin.
  munmap(grid, length);
  close(file_descriptor);

}

// -------------- MAIN PROGRAM --------------------------------------------------------------------

int main(int argc, char *argv[])
{

    // Check number of arguments for main function
    if (argc != 1)
    {
        cerr << "Usage: ./ATMO_retrieve" << endl;
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
    ArrayXd covariates = data.col(0).block(0,0,4999,1);
    ArrayXd observations = data.col(1).block(0,0,4999,1);
    ArrayXd uncertainties = data.col(2).block(0,0,4999,1);

    // Uniform Prior
    unsigned long Nparameters;  // Number of parameters for which prior distributions are defined

    int Ndimensions = 6;        // Number of free parameters (dimensions) of the problem
    vector<Prior*> ptrPriors(1);
    ArrayXd parametersMinima(Ndimensions);
    ArrayXd parametersMaxima(Ndimensions);
    parametersMinima <<  0,0,0,0,0,0;//0.5, 0.111, 0.333, 0.0;//, 0.044, 0.1;         // Minima values for the free parameters
    parametersMaxima <<0.999999,0.999999,0.999999,0.999999,0.999999,0.999999;// 0.773, 0.555, 0.956, 0.769;//, 0.272, 0.3;     // Maxima values for the free parameters

    UniformPrior uniformPrior(parametersMinima, parametersMaxima);
    ptrPriors[0] = &uniformPrior;

    string fullPathHyperParameters = outputPathPrefix + "hyperParametersUniform.txt";       // Print prior hyper parameters as output
    uniformPrior.writeHyperParametersToFile(fullPathHyperParameters);

    // SET UP GRID --------------------------------------------------------------------------------



    // --------------------------------------------------------------------------------------------
    // Set up the models for the inference problem
    ATMOModel model(covariates);      // ATMO function

    // Set up the likelihood function to be used
    NormalLikelihood likelihood(observations, uncertainties, model);

    // Set up the K-means clusterer using an Euclidean metric

    int minNclusters = 1;  // Minimum number of clusters
    int maxNclusters = 5; // Maximum number of clusters

    int Ntrials = 10;
    double relTolerance = 0.01;

    EuclideanMetric myMetric;
    KmeansClusterer kmeans(myMetric, minNclusters, maxNclusters, Ntrials, relTolerance);

    // Configure and start nested sampling inference -----

    bool printOnTheScreen = true;                  // Print results on the screen
    int initialNobjects = 100;                     // Initial number of live points
    int minNobjects = 100;                         // Minimum number of live points
    int maxNdrawAttempts = 20000;                  // Maximum number of attempts when trying to draw a new sampling point
    int NinitialIterationsWithoutClustering = 100; // The first N iterations, we assume that there is only 1 cluster
    int NiterationsWithSameClustering = 50;        // Clustering is only happening every N iterations.
    double initialEnlargementFraction =1.95;      // Fraction by which each axis in an ellipsoid has to be enlarged.
                                                            // It can be a number >= 0, where 0 means no enlargement.
    double shrinkingRate = 0.02;//0.02;                   // Exponent for remaining prior mass in ellipsoid enlargement fraction.
                                                            // It is a number between 0 and 1. The smaller the slower the shrinkage
                                                            // of the ellipsoids.

    double terminationFactor = 0.05;    // Termination factor for nested sampling process.

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
