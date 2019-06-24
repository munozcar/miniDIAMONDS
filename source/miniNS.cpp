
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include "miniNS.h"


// -------------- PRINCIPAL INFERENCE PROGRAM -----------------------------------------------------


// -------------- PROGRAM TO FIND EUCLIDEAN DISTANCE ----------------------------------------------
double EuclideanMetric::distance(RefArrayXd point1, RefArrayXd point2)
{
    return sqrt((point1-point2).square().sum());
}
// -------------- CLUSTERER -----------------------------------------------------------------------
Clusterer::Clusterer(Metric &metric)
: metric(metric)
{
}
// -------------- KMEANS CLUSTERER ----------------------------------------------------------------
KmeansClusterer::KmeansClusterer(Metric &metric, unsigned int minNclusters, unsigned int maxNclusters, unsigned int Ntrials, double relTolerance)
: Clusterer(metric),
  minNclusters(minNclusters),
  maxNclusters(maxNclusters),
  Ntrials(Ntrials),
  relTolerance(relTolerance)
{
    clock_t clockticks = clock();
    engine.seed(clockticks);
    assert(minNclusters <= maxNclusters);
}

KmeansClusterer::~KmeansClusterer()
{
}

void KmeansClusterer::chooseInitialClusterCenters(RefArrayXXd sample, RefArrayXXd centers, unsigned int Nclusters)
{
    unsigned int Npoints = sample.cols();
    uniform_real_distribution<> uniform01(0.0, 1.0);
    uniform_int_distribution<> uniform(0, Npoints-1);

    int randomPointIndex = uniform(engine);
    int k;
    double distanceToClosestCenter[Npoints];
    double sumOfDistancesToClosestCenters;
    double distance;
    double uniform01Number;
    double cumulativeDistance;
    centers.col(0) = sample.col(randomPointIndex);

    for (int n = 1; n < Nclusters; ++n)
    {
        sumOfDistancesToClosestCenters = 0.0;

        for (int k = 0; k < Npoints; ++k)
        {
            for (int j = 0; j < n; ++j)
            {
                distance = metric.distance(sample.col(k), centers.col(j));
                if (j == 0)
                {
                    distanceToClosestCenter[k] = distance;
                }
                else
                {
                    if (distance < distanceToClosestCenter[k])
                    {
                        distanceToClosestCenter[k] = distance;
                    }
                }
            }

            sumOfDistancesToClosestCenters += distanceToClosestCenter[k];
        }

        for (int k = 0; k < Npoints; ++k)
        {
            distanceToClosestCenter[k] /= sumOfDistancesToClosestCenters;
        }

        uniform01Number = uniform01(engine);
        cumulativeDistance = distanceToClosestCenter[0];
        k = 0;

        while (cumulativeDistance < uniform01Number)
        {
            k++;
            cumulativeDistance += distanceToClosestCenter[k];
        }

        centers.col(n) = sample.col(k);

    } // end loop of selecting initial cluster centers
}

bool KmeansClusterer::updateClusterCentersUntilConverged(RefArrayXXd sample, RefArrayXXd centers,
                                                         RefArrayXd clusterSizes, vector<int> &clusterIndices,
                                                         double &sumOfDistancesToClosestCenter, double relTolerance)
{
    unsigned int Npoints = sample.cols();
    unsigned int Ndimensions = sample.rows();
    unsigned int Nclusters = centers.cols();
    ArrayXXd updatedCenters = ArrayXXd::Zero(Ndimensions, Nclusters);   // coordinates of each of the new cluster centers

    bool stopIterations = false;
    bool convergenceReached;
    unsigned int indexOfClosestCenter;
    double oldSumOfDistances = 0.0;
    double newSumOfDistances = 0.0;
    double distanceToClosestCenter;
    double distance;

    while (!stopIterations)
    {
        clusterSizes.setZero();
        updatedCenters.setZero();

        for (int n = 0; n < Npoints; ++n)
        {
            distanceToClosestCenter = numeric_limits<double>::max();

            for (int i = 0; i < Nclusters; ++i)
            {
                distance = metric.distance(sample.col(n), centers.col(i));

                if (distance < distanceToClosestCenter)
                {
                    indexOfClosestCenter = i;
                    distanceToClosestCenter = distance;
                }
            }
            newSumOfDistances += distanceToClosestCenter;
            updatedCenters.col(indexOfClosestCenter) += sample.col(n);
            clusterSizes(indexOfClosestCenter) += 1;
            clusterIndices[n] = indexOfClosestCenter;
        }

        if (!(clusterSizes > 1).all())
        {
            convergenceReached = false;
            return convergenceReached;
        }

        updatedCenters.rowwise() /= clusterSizes.transpose();
        centers = updatedCenters;
        if (oldSumOfDistances == 0.0)
        {
            oldSumOfDistances = newSumOfDistances;
            newSumOfDistances = 0.0;
        }
        else
        {
            if (fabs(newSumOfDistances - oldSumOfDistances) / oldSumOfDistances < relTolerance)
            {
                sumOfDistancesToClosestCenter = newSumOfDistances;      // will be returned to user
                stopIterations = true;
            }
            else
            {
                oldSumOfDistances = newSumOfDistances;
                newSumOfDistances = 0.0;
            }
        }
    }  // end k-means center-updating loop
    convergenceReached = true;
    return convergenceReached;
}

double KmeansClusterer::evaluateBICvalue(RefArrayXXd sample, RefArrayXXd centers,
                                         RefArrayXd clusterSizes, vector<int> &clusterIndices)
{
    unsigned int Npoints = sample.cols();
    unsigned int Ndimensions = sample.rows();
    unsigned int Nclusters = centers.cols();

    ArrayXd intraClusterVariances(Nclusters);
    intraClusterVariances.setZero();

    for (int n = 0; n < Npoints; ++n)
    {
        intraClusterVariances(clusterIndices[n]) += metric.distance(sample.col(n), centers.col(clusterIndices[n]));
    }

    intraClusterVariances /= (clusterSizes-1);

    ArrayXd clusterPriors = clusterSizes / double(Npoints);

    double logLikelihood = 0.0;
    int cluster;

    for (int n = 0; n < Npoints; ++n)
    {
        cluster = clusterIndices[n];
        logLikelihood +=   log(clusterPriors(cluster))
                         - Ndimensions / 2.0 * log(intraClusterVariances(cluster))
                         - metric.distance(sample.col(n), centers.col(cluster)) / 2.0 / intraClusterVariances(cluster);
    }

    unsigned int NfreeParameters = Ndimensions * Nclusters + Nclusters + Nclusters-1;

    double BICvalue = -2.0 * logLikelihood + NfreeParameters * log(Npoints);

    return BICvalue;
}

int KmeansClusterer::cluster(RefArrayXXd sample, vector<int> &optimalClusterIndices, vector<int> &optimalClusterSizes)
{
    bool convergedSuccessfully;
    unsigned int Npoints = sample.cols();
    unsigned int Ndimensions = sample.rows();
    unsigned int optimalNclusters;
    double bestBICvalue = numeric_limits<double>::max();
    double BICvalue;
    double sumOfDistancesToClosestCenter;
    double bestSumOfDistancesToClosestCenter = numeric_limits<double>::max();
    vector<int> clusterIndices(Npoints);                    // For each point the index of the cluster to ...
    vector<int> bestClusterIndices(Npoints);                // ... which it belongs
    ArrayXd clusterSizes;                                   // Not vector<int> because will be used in Eigen array expressions
    ArrayXd bestClusterSizes;
    ArrayXXd centers;
    ArrayXXd bestCenters;

    for (unsigned int Nclusters = minNclusters; Nclusters <= maxNclusters; ++Nclusters)
    {
        centers = ArrayXXd::Zero(Ndimensions, Nclusters);          // coordinates of each of the old cluster centers
        bestCenters = ArrayXXd::Zero(Ndimensions, Nclusters);      // coordinates of the best centers (over all trials)
        clusterSizes = ArrayXd::Zero(Nclusters);                    // # of points belonging to each cluster...
        bestClusterSizes = ArrayXd::Zero(Nclusters);                // ... 'double', to avoid casting problems.

        bestSumOfDistancesToClosestCenter = numeric_limits<double>::max();


        for (int m = 0; m < Ntrials; ++m)
        {
            chooseInitialClusterCenters(sample, centers, Nclusters);
            convergedSuccessfully = updateClusterCentersUntilConverged(sample, centers, clusterSizes, clusterIndices,
                                                                       sumOfDistancesToClosestCenter, relTolerance);

            if (!convergedSuccessfully) continue;

            if (sumOfDistancesToClosestCenter < bestSumOfDistancesToClosestCenter)
            {
                bestSumOfDistancesToClosestCenter = sumOfDistancesToClosestCenter;
                bestCenters = centers;
                bestClusterIndices = clusterIndices;
                bestClusterSizes = clusterSizes;
            }
        } // end loop over Ntrials to determine the best clustering trying different initial centers

        if (maxNclusters - minNclusters > 1)
        {
            BICvalue = evaluateBICvalue(sample, bestCenters, bestClusterSizes, bestClusterIndices);

            if (BICvalue < bestBICvalue)
            {
                bestBICvalue = BICvalue;
                optimalNclusters = Nclusters;
                optimalClusterIndices = bestClusterIndices;
                optimalClusterSizes.resize(Nclusters);
                for (int n = 0; n < Nclusters; ++n)
                {
                    optimalClusterSizes[n] = bestClusterSizes(n);
                }
            }
        }
        else
        {
            optimalNclusters = Nclusters;
            optimalClusterIndices = bestClusterIndices;
            optimalClusterSizes.resize(Nclusters);
            for (int n = 0; n < Nclusters; ++n)
            {
                optimalClusterSizes[n] = bestClusterSizes(n);
            }
        }
    } // end loop over Nclusters
    return optimalNclusters;
}

// -------------- FUNCTIONS -----------------------------------------------------------------------
void Functions::lorentzProfile(RefArrayXd predictions, const RefArrayXd covariates,
                               const double centroid, const double amplitude, const double gamma)
{
    predictions = (amplitude*amplitude)/((covariates-centroid)*(covariates-centroid) + (gamma/2.)*(gamma/2.));
}

void Functions::modeProfile(RefArrayXd predictions, const RefArrayXd covariates,
                               const double centroid, const double height, const double linewidth)
{
    predictions = height/(1.0 + (4.0*(covariates-centroid).square()/(linewidth*linewidth)));
}

void Functions::modeProfileWithAmplitude(RefArrayXd predictions, const RefArrayXd covariates,
                               const double centroid, const double amplitude, const double linewidth)
{
    predictions = amplitude*amplitude/(Functions::PI * linewidth)/(1.0 + (4.0*(covariates-centroid).square()/(linewidth*linewidth)));
}

void Functions::modeProfileSinc(RefArrayXd predictions, const RefArrayXd covariates,
                               const double centroid, const double height, const double resolution)
{
    ArrayXd sincFunctionArgument = Functions::PI*(covariates - centroid)/resolution;
    ArrayXd sincFunction = sincFunctionArgument.sin() / sincFunctionArgument;


    // Multiply the profile by the height in the PSD

    predictions = height*sincFunction.square();
}

double Functions::logGaussProfile(const double covariate, const double mu,
                                  const double sigma, const double amplitude)
{
    const double prefactor = log(amplitude) - 0.5 * log(2*Functions::PI) - log(sigma);

    return prefactor - 0.5 * (covariate - mu) * (covariate - mu) / (sigma * sigma);

}

void Functions::logGaussProfile(RefArrayXd predictions, const RefArrayXd covariates, const double mu,
                                const double sigma, const double amplitude)
{
    const double prefactor = log(amplitude) - 0.5 * log(2*Functions::PI) - log(sigma);
    predictions = prefactor - 0.5 * (covariates - mu) * (covariates - mu) / (sigma * sigma);

}

double Functions::logGaussLikelihood(const RefArrayXd observations, const RefArrayXd predictions, const RefArrayXd uncertainties)
{
    if ((observations.size() != predictions.size()) || (observations.size() != uncertainties.size()))
    {
        cerr << "Array dimensions do not match. Quitting program." << endl;
        exit(EXIT_FAILURE);
    }

    ArrayXd delta;
    ArrayXd lambda0;
    ArrayXd lambda;

    delta = ((observations - predictions)*(observations - predictions)) / (uncertainties*uncertainties);
    lambda0 = -1.*log(sqrt(2.*PI) * uncertainties);
    lambda = lambda0 -0.5*delta;

    return lambda.sum();

}

void Functions::clusterCovariance(RefArrayXXd const clusterSample, RefArrayXXd covarianceMatrix,
                                  RefArrayXd centerCoordinates)
{
    int Ndimensions = clusterSample.rows();
    int Npoints = clusterSample.cols();
    covarianceMatrix.resize(Ndimensions, Ndimensions);
    centerCoordinates.resize(Ndimensions);

    for (int i=0; i < Ndimensions; i++)
    {
        centerCoordinates(i) = (clusterSample.row(i).sum())/Npoints;
    }

    double biasFactor = 1./(Npoints-1);

    for (int i=0; i < Ndimensions; i++)
    {
        for (int j=0; j < Ndimensions; j++)
        {
            covarianceMatrix(i,j) = biasFactor * ((clusterSample.row(i) - centerCoordinates(i))*(clusterSample.row(j) - centerCoordinates(j))).sum();
        }
    }

}

bool Functions::selfAdjointMatrixDecomposition(RefArrayXXd const covarianceMatrix, RefArrayXd eigenvalues,
                                               RefArrayXXd eigenvectorsMatrix)
{
    assert(covarianceMatrix.cols() == covarianceMatrix.rows());
    assert(eigenvalues.size() == covarianceMatrix.cols());
    assert(eigenvectorsMatrix.cols() == eigenvectorsMatrix.rows());
    assert(eigenvectorsMatrix.cols() == eigenvalues.size());

    SelfAdjointEigenSolver<MatrixXd> eigenSolver(covarianceMatrix.matrix());

    if (eigenSolver.info() != Success)
    {
        cout << "Covariance Matrix decomposition failed." << endl;
        cout << "Quitting program" << endl;
        return false;
    }

    eigenvalues = eigenSolver.eigenvalues();
    eigenvectorsMatrix = eigenSolver.eigenvectors();

    return true;
}

inline double Functions::product(const vector<double> &vec)
{
    return accumulate(vec.begin(), vec.end(), 1.0, multiplies<double>());
}

inline double Functions::sum(const vector<double> &vec)
{
    return accumulate(vec.begin(), vec.end(), 0.0);
}

double Functions::logExpSum(const double x, const double y)
{
    return (x >= y ? x + log(1.+exp(y-x)) : y + log(1.+exp(x-y)));
}

double Functions::logExpDifference(const double x, const double y)
{
    return (x >= y ? x + log(1.0 - exp(y-x)) : y + log(1.0 - exp(x-y)));
}

void Functions::topDownMerge(RefArrayXd array1, RefArrayXd arrayCopy1, RefArrayXd array2, RefArrayXd arrayCopy2, int beginIndex, int middleIndex, int endIndex)
{
    // Specify the initial indices of first and second-half of the input array

    int firstIndexFirstPart = beginIndex;
    int firstIndexSecondPart = middleIndex;


    // Do the sorting

    for (int i = beginIndex; i < endIndex; i++)
    {
        if ((firstIndexFirstPart < middleIndex) && (firstIndexSecondPart >= endIndex || (array1(firstIndexFirstPart) <= array1(firstIndexSecondPart))))
        {
            arrayCopy1(i) = array1(firstIndexFirstPart);
            arrayCopy2(i) = array2(firstIndexFirstPart);
            firstIndexFirstPart++;      // Move to the next element
        }
        else
        {
            arrayCopy1(i) = array1(firstIndexSecondPart);
            arrayCopy2(i) = array2(firstIndexSecondPart);
            firstIndexSecondPart++;     // Move to the next element
        }
    }
}

void Functions::topDownSplitMerge(RefArrayXd array1, RefArrayXd arrayCopy1, RefArrayXd array2, RefArrayXd arrayCopy2, int beginIndex, int endIndex)
{

    if (endIndex - beginIndex < 2)
        return;

    int middleIndex = (endIndex + beginIndex) / 2;

    topDownSplitMerge(array1, arrayCopy1, array2, arrayCopy2, beginIndex, middleIndex);
    topDownSplitMerge(array1, arrayCopy1, array2, arrayCopy2, middleIndex, endIndex);

    topDownMerge(array1, arrayCopy1, array2, arrayCopy2, beginIndex, middleIndex, endIndex);

    int length = endIndex - beginIndex;
    array1.segment(beginIndex, length) = arrayCopy1.segment(beginIndex, length);
    array2.segment(beginIndex, length) = arrayCopy2.segment(beginIndex, length);
}

void Functions::topDownMergeSort(RefArrayXd array1, RefArrayXd array2)
{
    assert(array1.size() == array2.size());
    ArrayXd arrayCopy1 = array1;
    ArrayXd arrayCopy2 = array2;
    topDownSplitMerge(array1, arrayCopy1, array2, arrayCopy2, 0, arrayCopy1.size());
}

void Functions::sortElementsDouble(RefArrayXd array1, RefArrayXd array2)
{
    assert(array1.size() == array2.size());

    for (int i = 0; i < array1.size(); i++)
    {
        for (int j = 1; j < (array1.size()-i); j++)
        {
            if (array1(j-1) > array1(j))
            {
                SWAPDOUBLE(array1(j-1),array1(j));        // SWAP array1 elements in increasing order
                SWAPDOUBLE(array2(j-1),array2(j));        // SWAP array2 elements accordingly
            }
            else
                if (array1(j-1) == array1(j))
                    continue;
        }
    }
}

void Functions::sortElementsInt(vector<int> &array1, RefArrayXd array2)
{
    for (int i = 0; i < array1.size(); i++)
    {
        for (int j = 1; j < (array1.size()-i); j++)
        {
            if (array1[j-1] > array1[j])
            {
                SWAPINT(array1[j-1], array1[j]);         // SWAP array1 elements in increasing order
                SWAPDOUBLE(array2(j-1), array2(j));      // SWAP array2 elements accordingly
            }
            else
                if (array1[j-1] == array1[j])
                    continue;
        }
    }
}

vector<int> Functions::findArrayIndicesWithinBoundaries(RefArrayXd const array, double lowerBound, double upperBound)
{
    // At least one point is needed

    assert(array.size() >= 1);
    vector<int> arrayIndicesWithinBoundaries;

    if (lowerBound < upperBound)
    {
        for (int i = 0; i < array.size(); ++i)
        {
            if ((array(i) >= lowerBound) && (array(i) <= upperBound))
                arrayIndicesWithinBoundaries.push_back(i);
            else
                continue;
        }
    }

    return arrayIndicesWithinBoundaries;
}

int Functions::countArrayIndicesWithinBoundaries(RefArrayXd const array, double lowerBound, double upperBound)
{
    // At least one point is needed

    assert(array.size() >= 1);
    int binSize = 0;

    if (lowerBound < upperBound)
    {
        for (int i = 0; i < array.size(); ++i)
        {
            if ((array(i) >= lowerBound) && (array(i) <= upperBound))
                binSize++;
        }
    }

    return binSize;
}

ArrayXd Functions::cubicSplineInterpolation(RefArrayXd const observedAbscissa, RefArrayXd const observedOrdinate,
                                            RefArrayXd const interpolatedAbscissaUntruncated)
{
    int size = observedAbscissa.size();
    int interpolatedSize = interpolatedAbscissaUntruncated.size();

    assert(size >= 2);
    assert(interpolatedSize >= 1);
    assert(observedOrdinate.size() == size);
    assert(observedAbscissa(0) <= interpolatedAbscissaUntruncated(0));

    double largestObservedAbscissa = observedAbscissa(size-1);
    double largestInterpolatedAbscissa = interpolatedAbscissaUntruncated(interpolatedSize-1);
    ArrayXd interpolatedAbscissa;

    if (largestObservedAbscissa < largestInterpolatedAbscissa)
    {

        int extraSize = Functions::countArrayIndicesWithinBoundaries(interpolatedAbscissaUntruncated, largestObservedAbscissa, largestInterpolatedAbscissa);
        interpolatedSize = interpolatedSize - extraSize;
        interpolatedAbscissa = interpolatedAbscissaUntruncated.segment(0,interpolatedSize);
    }
    else
        interpolatedAbscissa = interpolatedAbscissaUntruncated;

    ArrayXd differenceOrdinate = observedOrdinate.segment(1,size-1) - observedOrdinate.segment(0,size-1);
    ArrayXd differenceAbscissa = observedAbscissa.segment(1,size-1) - observedAbscissa.segment(0,size-1);
    ArrayXd differenceAbscissa2 = observedAbscissa.segment(2,size-2) - observedAbscissa.segment(0,size-2);

    vector<double> secondDerivatives(size);
    vector<double> partialSolution(size-1);
    secondDerivatives[0] = 0.0;
    partialSolution[0] = 0.0;

    ArrayXd sigma = differenceAbscissa.segment(0,size-2)/differenceAbscissa2.segment(0,size-2);
    double beta;

    for (int i = 1; i < size-1; ++i)
    {
        beta = sigma(i-1) * secondDerivatives[i-1] + 2.0;
        secondDerivatives[i] = (sigma(i-1) - 1.0)/beta;
        partialSolution[i] = differenceOrdinate(i)/differenceAbscissa(i) - differenceOrdinate(i-1)/differenceAbscissa(i-1);
        partialSolution[i] = (6.0*partialSolution[i]/differenceAbscissa2(i-1)-sigma(i-1)*partialSolution[i-1])/beta;
    }

    secondDerivatives[size-1] = 0.0;

    for (int k = (size-2); k >= 0; --k)
    {
        secondDerivatives[k] = secondDerivatives[k]*secondDerivatives[k+1]+partialSolution[k];
    }

    ArrayXd interpolatedOrdinate = ArrayXd::Zero(interpolatedSize);
    ArrayXd remainingInterpolatedAbscissa = interpolatedAbscissa;       // The remaining part of the array of interpolated abscissa
    int cumulatedBinSize = 0;                                           // The cumulated number of interpolated points from the beginning
    int i = 0;                                                          // Bin counter

    while ((i < size-1) && (cumulatedBinSize < interpolatedSize))
    {

        double lowerAbscissa = observedAbscissa(i);
        double upperAbscissa = observedAbscissa(i+1);

        int binSize = Functions::countArrayIndicesWithinBoundaries(remainingInterpolatedAbscissa, lowerAbscissa, upperAbscissa);

        if (binSize > 0)
        {
            double lowerOrdinate = observedOrdinate(i);
            double upperOrdinate = observedOrdinate(i+1);
            double denominator = differenceAbscissa(i);
            double upperSecondDerivative = secondDerivatives[i+1];
            double lowerSecondDerivative = secondDerivatives[i];
            ArrayXd interpolatedAbscissaInCurrentBin = remainingInterpolatedAbscissa.segment(0, binSize);
            ArrayXd interpolatedOrdinateInCurrentBin = ArrayXd::Zero(binSize);

            ArrayXd a = (upperAbscissa - interpolatedAbscissaInCurrentBin) / denominator;
            ArrayXd b = 1.0 - a;
            ArrayXd c = (1.0/6.0) * (a.cube() - a)*denominator*denominator;
            ArrayXd d = (1.0/6.0) * (b.cube() - b)*denominator*denominator;
            interpolatedOrdinateInCurrentBin = a*lowerOrdinate + b*upperOrdinate + c*lowerSecondDerivative + d*upperSecondDerivative;

            interpolatedOrdinate.segment(cumulatedBinSize, binSize) = interpolatedOrdinateInCurrentBin;

            int currentRemainingSize = interpolatedSize - cumulatedBinSize;
            remainingInterpolatedAbscissa.resize(currentRemainingSize - binSize);
            cumulatedBinSize += binSize;
            remainingInterpolatedAbscissa = interpolatedAbscissa.segment(cumulatedBinSize, interpolatedSize-cumulatedBinSize);
        }
        ++i;
    }

    return interpolatedOrdinate;
}

// -------------- ELLIPSOID CONSTRUCTOR -----------------------------------------------------------
Ellipsoid::Ellipsoid(RefArrayXXd sample, const double enlargementFraction)
: sample(sample),
  sampleSize(sample.cols()),
  Ndimensions(sample.rows()),
  uniform(0.0, 1.0),
  normal(0.0, 1.0)
{
    clock_t clockticks = clock();
    engine.seed(clockticks);

    originalEigenvalues.resize(Ndimensions);
    enlargedEigenvalues.resize(Ndimensions);
    centerCoordinates.resize(Ndimensions);
    eigenvectors.resize(Ndimensions, Ndimensions);
    covarianceMatrix.resize(Ndimensions, Ndimensions);

    Functions::clusterCovariance(sample, covarianceMatrix, centerCoordinates);

    bool covarianceDecompositionIsSuccessful = Functions::selfAdjointMatrixDecomposition(covarianceMatrix, originalEigenvalues, eigenvectors);
    if (!covarianceDecompositionIsSuccessful) abort();

    resetEnlargementFraction(enlargementFraction);
}

Ellipsoid::~Ellipsoid()
{

}

void Ellipsoid::resetEnlargementFraction(const double newEnlargementFraction)
{
    this->enlargementFraction = newEnlargementFraction;

    enlargedEigenvalues = originalEigenvalues.sqrt() + enlargementFraction * originalEigenvalues.sqrt();
    enlargedEigenvalues = enlargedEigenvalues.square();

    hyperVolume = enlargedEigenvalues.sqrt().prod();
    covarianceMatrix = eigenvectors.matrix() * enlargedEigenvalues.matrix().asDiagonal() * eigenvectors.matrix().transpose();
}

bool Ellipsoid::overlapsWith(Ellipsoid ellipsoid, bool &ellipsoidMatrixDecompositionIsSuccessful)
{

    MatrixXd T1 = MatrixXd::Identity(Ndimensions+1,Ndimensions+1);
    MatrixXd T2 = MatrixXd::Identity(Ndimensions+1,Ndimensions+1);

    T1.bottomLeftCorner(1,Ndimensions) = (-1.0) * centerCoordinates.transpose();
    T2.bottomLeftCorner(1,Ndimensions) = (-1.0) * ellipsoid.getCenterCoordinates().transpose();

    MatrixXd A = MatrixXd::Zero(Ndimensions+1,Ndimensions+1);
    MatrixXd B = A;

    A(Ndimensions,Ndimensions) = -1;
    B(Ndimensions,Ndimensions) = -1;

    A.topLeftCorner(Ndimensions,Ndimensions) = covarianceMatrix.matrix().inverse();
    B.topLeftCorner(Ndimensions,Ndimensions) = ellipsoid.getCovarianceMatrix().matrix().inverse();

    MatrixXd AT = T1*A*T1.transpose();        // Translating to ellipsoid center
    MatrixXd BT = T2*B*T2.transpose();        // Translating to ellipsoid center

    MatrixXd C = AT.inverse() * BT;
    MatrixXcd CC(Ndimensions+1,Ndimensions+1);

    CC.imag() = MatrixXd::Zero(Ndimensions+1,Ndimensions+1);
    CC.real() = C;

    ComplexEigenSolver<MatrixXcd> eigenSolver(CC);

    if (eigenSolver.info() != Success)
    {
        ellipsoidMatrixDecompositionIsSuccessful = false;
    }

    MatrixXcd E = eigenSolver.eigenvalues();
    MatrixXcd V = eigenSolver.eigenvectors();

    bool ellipsoidsDoOverlap = false;       // Assume no overlap in the beginning
    double pointA;                          // Point laying in this ellipsoid
    double pointB;                          // Point laying in the other ellipsoid

    for (int i = 0; i < Ndimensions+1; i++)
    {

        if (V(Ndimensions,i).real() == 0)
        {
            continue;
        }
        else if (E(i).imag() != 0)
            {
                V.col(i) = V.col(i).array() * (V.conjugate())(Ndimensions,i);      // Multiply eigenvector by complex conjugate of last element
                V.col(i) = V.col(i).array() / V(Ndimensions,i).real();             // Normalize eigenvector to last component value
                pointA = V.col(i).transpose().real() * AT * V.col(i).real();       // Evaluate point from this ellipsoid
                pointB = V.col(i).transpose().real() * BT * V.col(i).real();       // Evaluate point from the other ellipsoid

                if ((pointA <= 0) && (pointB <= 0))
                {
                    ellipsoidsDoOverlap = true;            // Exit if ellipsoidsDoOverlap is found
                    break;
                }
            }
    }
    return ellipsoidsDoOverlap;
}

bool Ellipsoid::containsPoint(const RefArrayXd pointCoordinates)
{

    MatrixXd T = MatrixXd::Identity(Ndimensions+1,Ndimensions+1);

    T.bottomLeftCorner(1,Ndimensions) = (-1.) * centerCoordinates.transpose();

    MatrixXd A = MatrixXd::Zero(Ndimensions+1,Ndimensions+1);
    A(Ndimensions,Ndimensions) = -1;

    MatrixXd C = MatrixXd::Zero(Ndimensions, Ndimensions);

    C =  eigenvectors.matrix() * enlargedEigenvalues.matrix().asDiagonal()
                               * eigenvectors.matrix().transpose();
    A.topLeftCorner(Ndimensions,Ndimensions) = C.inverse();

    MatrixXd AT = T * A * T.transpose();

    VectorXd X(Ndimensions+1);
    X.head(Ndimensions) = pointCoordinates.matrix();
    X(Ndimensions) = 1;

    bool pointBelongsToThisEllipsoid;

    if (X.transpose() * AT * X <= 0)
    {
        pointBelongsToThisEllipsoid = true;
    }
    else
    {
        pointBelongsToThisEllipsoid = false;
    }

    return pointBelongsToThisEllipsoid;
}

void Ellipsoid::drawPoint(RefArrayXd drawnPoint)
{
    do
    {
        for (int i = 0; i < Ndimensions; i++)
        {
            drawnPoint(i) = normal(engine);
        }
    }
    while ((drawnPoint == 0.0).all());    // Repeat sampling if point falls in origin

    drawnPoint = drawnPoint / drawnPoint.matrix().norm();
    drawnPoint = pow(uniform(engine), 1./Ndimensions) * drawnPoint;

    MatrixXd D = enlargedEigenvalues.sqrt().matrix().asDiagonal();
    MatrixXd T = eigenvectors.matrix() * D;

    drawnPoint = (T * drawnPoint.matrix()) + centerCoordinates.matrix();
}

ArrayXd Ellipsoid::getCenterCoordinates()
{
    return centerCoordinates;
}

ArrayXd Ellipsoid::getEigenvalues()
{
    return enlargedEigenvalues;
}

ArrayXXd Ellipsoid::getSample()
{
    return sample;
}

ArrayXXd Ellipsoid::getCovarianceMatrix()
{
    return covarianceMatrix;
}

ArrayXXd Ellipsoid::getEigenvectors()
{
    return eigenvectors;
}

int Ellipsoid::getSampleSize()
{
    return sampleSize;
}

double Ellipsoid::getHyperVolume()
{
    return hyperVolume;
}

double Ellipsoid::getEnlargementFraction()
{
    return enlargementFraction;
}

// -------------- MODEL CONSTRUCTOR ---------------------------------------------------------------
Model::Model(const RefArrayXd covariates)
: covariates(covariates)
{

}

Model::~Model()
{

}

ArrayXd Model::getCovariates()
{
    return covariates;
}

int Model::getNparameters()
{
    return Nparameters;
}

// -------------- LIKELIHOOD CONSTRUCTOR ----------------------------------------------------------
Likelihood::Likelihood(const RefArrayXd observations, Model &model)
: observations(observations),
  model(model)
{

} // END Likelihood::Likelihood()

Likelihood::~Likelihood()
{

} // END Likelihood::Likelihood()

ArrayXd Likelihood::getObservations()
{
    return observations;
} // END Likelihood::getObservations()

// -------------- NORMAL LIKELIHOOD CONSTRUCTOR ---------------------------------------------------
NormalLikelihood::NormalLikelihood(const RefArrayXd observations, const RefArrayXd uncertainties, Model &model)
: Likelihood(observations, model),
  uncertainties(uncertainties)
{
    assert(observations.size() || uncertainties.size());
}

NormalLikelihood::~NormalLikelihood()
{

}

ArrayXd NormalLikelihood::getUncertainties()
{
    return uncertainties;
}

double NormalLikelihood::logValue(RefArrayXd modelParameters)
{
    ArrayXd predictions;
    ArrayXd lambda;
    ArrayXd lambda0;

    predictions.resize(observations.size());
    predictions.setZero();
    model.predict(predictions, modelParameters);

    lambda0 = -0.5 * observations.size() * log(2.0*Functions::PI) -1.0 * uncertainties.log();
    lambda = lambda0 - 0.5 * ((observations - predictions)*(observations - predictions)) / (uncertainties*uncertainties);

    return lambda.sum();
}

// -------------- FILE READING PROGRAM -----------------------------------------------------------
void File::openInputFile(ifstream &inputFile, string inputFileName)
{
    inputFile.open(inputFileName.c_str());
    if (!inputFile.good())
    {
        cerr << "Error opening input file " << inputFileName << endl;
        exit(EXIT_FAILURE);
    }
}

void File::openOutputFile(ofstream &outputFile, string outputFileName)
{
    outputFile.open(outputFileName.c_str());
    if (!outputFile.good())
    {
        cerr << "Error opening input file " << outputFileName << endl;
        exit(EXIT_FAILURE);
    }
}

ArrayXXd File::arrayXXdFromFile(ifstream &inputFile, const unsigned long Nrows, const int Ncols, char separator, char commentChar)
{
    string line;
    unsigned long iRow = 0;
    int iCol = 0;
    ArrayXXd array = ArrayXXd::Zero(Nrows, Ncols);

    while(!inputFile.eof())
    {
        getline(inputFile, line);

        if (line[0] == commentChar) continue;

        const std::string whitespace = " \t";
        const auto strBegin = line.find_first_not_of(whitespace);
        if (strBegin == string::npos) continue;

        if (iRow > Nrows-1)
        {
            cerr << "Error: numbers of rows in file exceeds " << Nrows << endl;
            exit(EXIT_FAILURE);
        }

        vector<string> tokens;
        string::size_type begin = line.find_first_not_of(separator);
        string::size_type end = line.find_first_of(separator, begin);
        while (begin != string::npos || end != string::npos)
        {
            tokens.push_back(line.substr(begin, end-begin));
            begin = line.find_first_not_of(separator, end);
            end = line.find_first_of(separator, begin);
        }

        if (tokens.size() != Ncols)
        {
            cerr << "Error on row " << iRow << ": number of tokens != "
                 << Ncols << endl;
            exit(EXIT_FAILURE);
        }

        iCol = 0;
        for (auto token : tokens)
        {
            istringstream mystream(token);
            double value;
            if (!(mystream >> value))
            {
                cerr << "Error on row " << iRow << ". Can't convert "
                     << token << " to a number" << endl;
                exit(EXIT_FAILURE);
            }
            else
            {
                array(iRow, iCol) = value;
                iCol++;
            }
        }

        iRow++;
    }

    if (iRow < Nrows)
    {
        cerr << "Warning: too few lines in the file: " << iRow << " < " << Nrows << endl;
    }

    return array;
}

vector<string> File::vectorStringFromFile(ifstream &inputFile, const unsigned long Nrows, char commentChar)
{
    string line;
    unsigned long iRow = 0;
    vector<string> vectorString(Nrows);

    while(!inputFile.eof())
    {
        getline(inputFile, line);
        if (line[0] == commentChar) continue;

        const std::string whitespace = " \t";
        const auto strBegin = line.find_first_not_of(whitespace);
        if (strBegin == string::npos) continue;

        if (iRow > Nrows-1)
        {
            cerr << "Error: numbers of rows in file exceeds " << Nrows << endl;
            exit(EXIT_FAILURE);
        }

        vectorString[iRow] = line;
        iRow++;
    }
    if (iRow < Nrows)
    {
        cerr << "Warning: too few lines in the file: " << iRow << " < " << Nrows << endl;
    }
    return vectorString;
}

void File::arrayXXdToFile(ofstream &outputFile, RefArrayXXd array, string separator, string terminator)
{
    for (ptrdiff_t i = 0; i < array.rows(); ++i)
    {
        for (ptrdiff_t j = 0; j < array.cols()-1; ++j)
        {
            outputFile << array(i,j) << separator;
        }
        outputFile << array(i,array.cols()-1) << terminator;
    }
}

void File::twoArrayXdToFile(ofstream &outputFile, RefArrayXd array1, RefArrayXd array2, string separator, string terminator)
{
    assert(array1.size() == array2.size());

    for (ptrdiff_t i = 0; i < array1.rows(); ++i)
    {
        outputFile << array1(i) << separator << array2(i) << terminator;
    }
}

void File::arrayXdToFile(ofstream &outputFile, RefArrayXd array, string terminator)
{
    for (ptrdiff_t i = 0; i < array.size(); ++i)
    {
        outputFile << array(i) << terminator;
    }
}

void File::arrayXXdRowsToFiles(RefArrayXXd array, string fullPathPrefix, string fileExtension, string terminator)
{
    int Nrows = array.rows();
    assert(Nrows > 0);

    for (int i = 0; i < Nrows; i++)
    {

        ostringstream numberString;
        numberString << setfill('0') << setw(3) << i;
        string fullPath = fullPathPrefix + numberString.str() + fileExtension;

        ofstream outputFile;
        File::openOutputFile(outputFile, fullPath);
        outputFile << setiosflags(ios::scientific) << setprecision(9);

        ArrayXd oneRow = array.row(i);
        File::arrayXdToFile(outputFile, oneRow, terminator);
        outputFile.close();
    }
}

void File::sniffFile(ifstream &inputFile, unsigned long &Nrows, int &Ncols, char separator, char commentChar)
{
    string line;
    int iRow = 0;

    while(!inputFile.eof())
    {
        getline(inputFile, line);

        if (line[0] == commentChar) continue;

        const std::string whitespace = " \t";
        const auto strBegin = line.find_first_not_of(whitespace);
        if (strBegin == string::npos) continue;

        if (iRow == 0)
        {
            vector<string> tokens;
            string::size_type begin = line.find_first_not_of(separator);
            string::size_type end = line.find_first_of(separator, begin);
            while (begin != string::npos || end != string::npos)
            {
                tokens.push_back(line.substr(begin, end-begin));
                begin = line.find_first_not_of(separator, end);
                end = line.find_first_of(separator, begin);
            }

            Ncols = tokens.size();
        }

        iRow++;
    }

    Nrows = iRow;

    inputFile.clear();
    inputFile.seekg(ios::beg);
}

// -------------- PRIOR CONSTRUCTOR ---------------------------------------------------------------
Prior::Prior(const int Ndimensions)
: minusInfinity(numeric_limits<double>::lowest()),
  Ndimensions(Ndimensions)
{
    clock_t clockticks = clock();
    engine.seed(clockticks);
}

Prior::~Prior()
{

}

int Prior::getNdimensions()
{
    return Ndimensions;
}

// -------------- UNIFORM PRIOR CONSTRUCTOR -------------------------------------------------------
UniformPrior::UniformPrior(const RefArrayXd minima, const RefArrayXd maxima)
: Prior(minima.size()),
  uniform(0.0,1.0),
  minima(minima),
  maxima(maxima)
{
    assert (minima.size() == maxima.size());

    if ( (minima >= maxima).any() )
    {
        cerr << "Uniform Prior hyper parameters are not correctly typeset." << endl;
        exit(EXIT_FAILURE);
    }
}

UniformPrior::~UniformPrior()
{

}

ArrayXd UniformPrior::getMinima()
{
    return minima;
}

ArrayXd UniformPrior::getMaxima()
{
    return maxima;
}

double UniformPrior::logDensity(RefArrayXd const x, const bool includeConstantTerm)
{
    double logDens;
    if ((x < minima).any() | (x > maxima).any())
    {
        logDens = minusInfinity;
        return logDens;
    }
    else
    {
        logDens = 0.0;
    }

    if (includeConstantTerm)
    {
        logDens += (-1.0) * (maxima - minima).log().sum();
    }
    return logDens;
}

bool UniformPrior::drawnPointIsAccepted(RefArrayXd const drawnPoint)
{
    if (logDensity(drawnPoint) != minusInfinity)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void UniformPrior::draw(RefArrayXXd drawnSample)
{
    int Npoints = drawnSample.cols();

    for (int i = 0; i < Ndimensions; i++)
    {
        for (int j = 0; j < Npoints; j++)
        {
            drawnSample(i,j) = uniform(engine)*(maxima(i)-minima(i)) + minima(i);
        }
    }

}

void UniformPrior::drawWithConstraint(RefArrayXd drawnPoint, Likelihood &likelihood)
{
    double logLikelihood;
    double logLikelihoodConstraint = likelihood.logValue(drawnPoint);

    do
    {
        for (int i = 0; i < Ndimensions; i++)
            {
                drawnPoint(i) = uniform(engine)*(maxima(i) - minima(i)) + minima(i);
            }

        logLikelihood = likelihood.logValue(drawnPoint);
    }
    while (logLikelihood <= logLikelihoodConstraint);
}

void UniformPrior::writeHyperParametersToFile(string fullPath)
{
    ofstream outputFile;
    File::openOutputFile(outputFile, fullPath);
    outputFile << "# Hyper parameters used for setting up uniform priors." << endl;
    outputFile << "# Each line corresponds to a different free parameter (coordinate)." << endl;
    outputFile << "# Column #1: Minima (lower boundaries)" << endl;
    outputFile << "# Column #2: Maxima (upper boundaries)" << endl;
    File::twoArrayXdToFile(outputFile, minima, maxima);
    outputFile.close();
}

// -------------- PROGRAM TO REDUCE LIVE POINTS ---------------------------------------------------
LivePointsReducer::LivePointsReducer(NestedSampler &nestedSampler)
: nestedSampler(nestedSampler)
{
}

LivePointsReducer::~LivePointsReducer()
{
}

vector<int> LivePointsReducer::findIndicesOfLivePointsToRemove(mt19937 engine)
{

    NlivePointsToRemove = NlivePointsAtCurrentIteration - updatedNlivePoints;

    vector<int> indicesOfLivePointsToRemove;

    if (NlivePointsToRemove > 0)
    {

        for (int m = 0; m < NlivePointsToRemove; ++m)
        {

            uniform_int_distribution<int> discreteUniform(0, NlivePointsAtCurrentIteration-1);

            indicesOfLivePointsToRemove.push_back(discreteUniform(engine));

            --NlivePointsAtCurrentIteration;
        }
    }
    return indicesOfLivePointsToRemove;
}

int LivePointsReducer::getNlivePointsToRemove()
{
    return NlivePointsToRemove;
}

// -------------- NESTED SAMPLING ALGORITHM -------------------------------------------------------
NestedSampler::NestedSampler(const bool printOnTheScreen, const int initialNlivePoints, const int minNlivePoints, vector<Prior*> ptrPriors,
                             Likelihood &likelihood, Metric &metric, Clusterer &clusterer)
: ptrPriors(ptrPriors),
  likelihood(likelihood),
  metric(metric),
  clusterer(clusterer),
  printOnTheScreen(printOnTheScreen),
  NlivePoints(initialNlivePoints),
  minNlivePoints(minNlivePoints),
  logCumulatedPriorMass(numeric_limits<double>::lowest()),
  logRemainingPriorMass(0.0),
  ratioOfRemainderToCurrentEvidence(numeric_limits<double>::max()),
  Niterations(0),
  updatedNlivePoints(initialNlivePoints),
  initialNlivePoints(initialNlivePoints),
  informationGain(0.0),
  logEvidence(numeric_limits<double>::lowest())
{
    clock_t clockticks = clock();
    engine.seed(clockticks);

    Ndimensions = 0;

    for (int i = 0; i < ptrPriors.size(); i++)
    {
        Ndimensions += ptrPriors[i]->getNdimensions();
    }
}

NestedSampler::~NestedSampler()
{
}

void NestedSampler::run(LivePointsReducer &livePointsReducer, const int NinitialIterationsWithoutClustering,
                        const int NiterationsWithSameClustering, const int maxNdrawAttempts,
                        const double maxRatioOfRemainderToCurrentEvidence, string pathPrefix)
{
    int startTime = time(0);
    double logMeanLiveEvidence;
    terminationFactor = maxRatioOfRemainderToCurrentEvidence;
    outputPathPrefix = pathPrefix;

    if (printOnTheScreen)
    {
        cerr << "------------------------------------------------" << endl;
        cerr << " Bayesian Inference problem has " << Ndimensions << " dimensions." << endl;
        cerr << "------------------------------------------------" << endl;
        cerr << endl;
    }

    string fileName = "configuringParameters.txt";
    string fullPath = outputPathPrefix + fileName;
    File::openOutputFile(outputFile, fullPath);

    outputFile << "# List of configuring parameters used for the NSMC." << endl;
    outputFile << "# Row #1: Ndimensions" << endl;
    outputFile << "# Row #2: Initial(Maximum) NlivePoints" << endl;
    outputFile << "# Row #3: Minimum NlivePoints" << endl;
    outputFile << "# Row #4: NinitialIterationsWithoutClustering" << endl;
    outputFile << "# Row #5: NiterationsWithSameClustering" << endl;
    outputFile << "# Row #6: maxNdrawAttempts" << endl;
    outputFile << "# Row #7: terminationFactor" << endl;
    outputFile << "# Row #8: Niterations" << endl;
    outputFile << "# Row #9: Optimal Niterations" << endl;
    outputFile << "# Row #10: Final Nclusters" << endl;
    outputFile << "# Row #11: Final NlivePoints" << endl;
    outputFile << "# Row #12: Computational Time (seconds)" << endl;
    outputFile << Ndimensions << endl;
    outputFile << initialNlivePoints << endl;
    outputFile << minNlivePoints << endl;
    outputFile << NinitialIterationsWithoutClustering << endl;
    outputFile << NiterationsWithSameClustering << endl;
    outputFile << maxNdrawAttempts << endl;
    outputFile << terminationFactor << endl;

    uniform_int_distribution<int> discreteUniform(0, NlivePoints-1);

    if (printOnTheScreen)
    {
        cerr << "------------------------------------------------" << endl;
        cerr << " Doing initial sampling of parameter space..." << endl;
        cerr << "------------------------------------------------" << endl;
        cerr << endl;
    }

    nestedSample.resize(Ndimensions, NlivePoints);
    int beginIndex = 0;
    int NdimensionsOfCurrentPrior;
    ArrayXXd priorSample;

    for (int i = 0; i < ptrPriors.size(); i++)
    {

        NdimensionsOfCurrentPrior = ptrPriors[i]->getNdimensions();

        priorSample.resize(NdimensionsOfCurrentPrior, NlivePoints);
        ptrPriors[i]->draw(priorSample);

        nestedSample.block(beginIndex, 0, NdimensionsOfCurrentPrior, NlivePoints) = priorSample;
        beginIndex += NdimensionsOfCurrentPrior;
    }

    logLikelihood.resize(NlivePoints);

    for (int i = 0; i < NlivePoints; ++i)
    {
        logLikelihood(i) = likelihood.logValue(nestedSample.col(i));
    }

    double logWidthInPriorMass = log(1.0 - exp(-1.0/NlivePoints));                                             // X_0 - X_1    First width in prior mass
    logCumulatedPriorMass = Functions::logExpSum(logCumulatedPriorMass, logWidthInPriorMass);               // 1 - X_1
    logRemainingPriorMass = Functions::logExpDifference(logRemainingPriorMass, logWidthInPriorMass);        // X_1

    double logRemainingPriorMassRightBound = Functions::logExpDifference(log(2), logRemainingPriorMass);
    double logWidthInPriorMassRight = Functions::logExpDifference(logRemainingPriorMassRightBound,logRemainingPriorMass);

    logMaxLikelihoodOfLivePoints = logLikelihood.maxCoeff();

    unsigned int Nclusters = 0;
    vector<int> clusterIndices(NlivePoints);           // clusterIndices must have the same number of elements as the number of live points
    vector<int> clusterSizes;                       // The number of live points counted in each cluster is updated everytime one live point
                                                    // is removed from the sample.

    if (printOnTheScreen)
    {
        cerr << "-------------------------------" << endl;
        cerr << " Starting nested sampling...   " << endl;
        cerr << "-------------------------------" << endl;
        cerr << endl;
    }

    bool nestedSamplingShouldContinue = true;
    bool livePointsShouldBeReduced = (initialNlivePoints > minNlivePoints);       // Update live points only if required

    Niterations = 0;

    do
    {

        posteriorSample.conservativeResize(Ndimensions, Niterations + 1);
        logLikelihoodOfPosteriorSample.conservativeResize(Niterations + 1);
        logWeightOfPosteriorSample.conservativeResize(Niterations + 1);

        int indexOfLivePointWithWorstLikelihood;
        worstLiveLogLikelihood = logLikelihood.minCoeff(&indexOfLivePointWithWorstLikelihood);

        posteriorSample.col(Niterations) = nestedSample.col(indexOfLivePointWithWorstLikelihood);
        logLikelihoodOfPosteriorSample(Niterations) = worstLiveLogLikelihood;

        logMeanLikelihoodOfLivePoints = logLikelihood(0);

        for (int m = 1; m < NlivePoints; m++)
        {
            logMeanLikelihoodOfLivePoints = Functions::logExpSum(logMeanLikelihoodOfLivePoints, logLikelihood(m));
        }

        logMeanLikelihoodOfLivePoints -= log(NlivePoints);

        if ((Niterations % NiterationsWithSameClustering) == 0)
        {

            if (Niterations < NinitialIterationsWithoutClustering)
            {

                Nclusters = 1;
                clusterSizes.resize(1);
                clusterSizes[0] = NlivePoints;
                fill(clusterIndices.begin(), clusterIndices.end(), 0);
            }
            else
            {
                Nclusters = clusterer.cluster(nestedSample, clusterIndices, clusterSizes);
            }
        }

        int indexOfRandomlyChosenLivePoint = 0;

        if (NlivePoints > 1)
        {
            do
            {
                indexOfRandomlyChosenLivePoint = discreteUniform(engine);
            }
            while (indexOfRandomlyChosenLivePoint == indexOfLivePointWithWorstLikelihood);
        }
        ArrayXd drawnPoint = nestedSample.col(indexOfRandomlyChosenLivePoint);
        double logLikelihoodOfDrawnPoint = 0.0;
        bool newPointIsFound = drawWithConstraint(nestedSample, Nclusters, clusterIndices, clusterSizes,
                                                  drawnPoint, logLikelihoodOfDrawnPoint, maxNdrawAttempts);

        nestedSamplingShouldContinue = verifySamplerStatus();

        if (!nestedSamplingShouldContinue) break;

        if (!newPointIsFound)
        {
            nestedSamplingShouldContinue = false;
            cerr << "Can't find point with a better Likelihood." << endl;
            cerr << "Stopping the nested sampling loop prematurely." << endl;
            break;
        }

        nestedSample.col(indexOfLivePointWithWorstLikelihood) = drawnPoint;
        logLikelihood(indexOfLivePointWithWorstLikelihood) = logLikelihoodOfDrawnPoint;

        if (livePointsShouldBeReduced)
        {
            updatedNlivePoints = livePointsReducer.updateNlivePoints();

            if (updatedNlivePoints > NlivePoints)
            {
                cerr << "Something went wrong in the reduction of the live points." << endl;
                cerr << "The new number of live points is greater than the previous one." << endl;
                cerr << "Quitting program. " << endl;
                break;
            }

            livePointsShouldBeReduced = (updatedNlivePoints > minNlivePoints);

            if (updatedNlivePoints != NlivePoints)
            {
                vector<int> indicesOfLivePointsToRemove = livePointsReducer.findIndicesOfLivePointsToRemove(engine);

                removeLivePointsFromSample(indicesOfLivePointsToRemove, clusterIndices, clusterSizes);

                uniform_int_distribution<int> discreteUniform2(0, updatedNlivePoints-1);
                discreteUniform = discreteUniform2;
            }
        }

        NlivePointsPerIteration.push_back(NlivePoints);

        logMeanLiveEvidence = logMeanLikelihoodOfLivePoints + Niterations * (log(NlivePoints) - log(NlivePoints + 1));

        ratioOfRemainderToCurrentEvidence = exp(logMeanLiveEvidence - logEvidence);

        nestedSamplingShouldContinue = (ratioOfRemainderToCurrentEvidence > maxRatioOfRemainderToCurrentEvidence);

        double logStretchingFactor = Niterations*((1.0/NlivePoints) - (1.0/updatedNlivePoints));
        logWidthInPriorMass = logRemainingPriorMass + Functions::logExpDifference(0.0, logStretchingFactor - 1.0/updatedNlivePoints);  // X_i - X_(i+1)

        double logWidthInPriorMassLeft = logWidthInPriorMass;
        double logWeight = log(0.5) + Functions::logExpSum(logWidthInPriorMassLeft, logWidthInPriorMassRight);
        double logEvidenceContributionNew = logWeight + worstLiveLogLikelihood;

        logWeightOfPosteriorSample(Niterations) = logWeight;
        logWidthInPriorMassRight = logWidthInPriorMass;

        double logEvidenceNew = Functions::logExpSum(logEvidence, logEvidenceContributionNew);
        informationGain = exp(logEvidenceContributionNew - logEvidenceNew) * worstLiveLogLikelihood
                        + exp(logEvidence - logEvidenceNew) * (informationGain + logEvidence)
                        - logEvidenceNew;
        logEvidence = logEvidenceNew;

        if (printOnTheScreen)
        {
            if ((Niterations % 50) == 0)
            {
                cerr << "Nit: " << Niterations
                     << "   Ncl: " << Nclusters
                     << "   Nlive: " << NlivePoints
                     << "   CPM: " << exp(logCumulatedPriorMass)
                     << "   Ratio: " << ratioOfRemainderToCurrentEvidence
                     << "   log(E): " << logEvidence
                     << "   IG: " << informationGain
                     << endl;
            }
        }

        logCumulatedPriorMass = Functions::logExpSum(logCumulatedPriorMass, logWidthInPriorMass);
        logRemainingPriorMass = logStretchingFactor + logRemainingPriorMass - 1.0/updatedNlivePoints;

        NlivePoints = updatedNlivePoints;

        Niterations++;
    }
    while (nestedSamplingShouldContinue);

    unsigned int oldNpointsInPosterior = posteriorSample.cols();

    posteriorSample.conservativeResize(Ndimensions, oldNpointsInPosterior + NlivePoints);          // First make enough room
    posteriorSample.block(0, oldNpointsInPosterior, Ndimensions, NlivePoints) = nestedSample;      // Then copy the live sample to the posterior array
    logWeightOfPosteriorSample.conservativeResize(oldNpointsInPosterior + NlivePoints);
    logWeightOfPosteriorSample.segment(oldNpointsInPosterior, NlivePoints).fill(logRemainingPriorMass - log(NlivePoints));  // Check if the best condition to impose
    logLikelihoodOfPosteriorSample.conservativeResize(oldNpointsInPosterior + NlivePoints);
    logLikelihoodOfPosteriorSample.segment(oldNpointsInPosterior, NlivePoints) = logLikelihood;

    logEvidenceError = sqrt(fabs(informationGain)/NlivePoints);

    logEvidence = Functions::logExpSum(logMeanLiveEvidence, logEvidence);

    if (printOnTheScreen)
    {
        cerr << "------------------------------------------------" << endl;
        cerr << " Final log(E): " << logEvidence << " +/- " << logEvidenceError << endl;
        cerr << "------------------------------------------------" << endl;
    }

    printComputationalTime(startTime);

    outputFile << Niterations << endl;
    outputFile << static_cast<int>((NlivePoints*informationGain) + (NlivePoints*sqrt(Ndimensions*1.0))) << endl;
    outputFile << Nclusters << endl;
    outputFile << NlivePoints << endl;
    outputFile << computationalTime << endl;
}

void NestedSampler::removeLivePointsFromSample(const vector<int> &indicesOfLivePointsToRemove,
                                               vector<int> &clusterIndices, vector<int> &clusterSizes)
{
    int NlivePointsToRemove = indicesOfLivePointsToRemove.size();
    int NlivePointsAtCurrentIteration = clusterIndices.size();

    for (int m = 0; m < NlivePointsToRemove; ++m)
    {

        ArrayXd nestedSamplePerLivePointCopy(Ndimensions);
        nestedSamplePerLivePointCopy = nestedSample.col(NlivePointsAtCurrentIteration-1);
        nestedSample.col(NlivePointsAtCurrentIteration-1) = nestedSample.col(indicesOfLivePointsToRemove[m]);
        nestedSample.col(indicesOfLivePointsToRemove[m]) = nestedSamplePerLivePointCopy;
        nestedSample.conservativeResize(Ndimensions, NlivePointsAtCurrentIteration-1);

        double logLikelihoodCopy = logLikelihood(NlivePointsAtCurrentIteration-1);
        logLikelihood(NlivePointsAtCurrentIteration-1) = logLikelihood(indicesOfLivePointsToRemove[m]);
        logLikelihood(indicesOfLivePointsToRemove[m]) = logLikelihoodCopy;
        logLikelihood.conservativeResize(NlivePointsAtCurrentIteration-1);

        int clusterIndexCopy = clusterIndices[NlivePointsAtCurrentIteration-1];
        clusterIndices[NlivePointsAtCurrentIteration-1] = clusterIndices[indicesOfLivePointsToRemove[m]];
        --clusterSizes[clusterIndices[indicesOfLivePointsToRemove[m]]];
        clusterIndices[indicesOfLivePointsToRemove[m]] = clusterIndexCopy;
        clusterIndices.pop_back();

        --NlivePointsAtCurrentIteration;
    }
}

void NestedSampler::printComputationalTime(const double startTime)
{
    double endTime = time(0);
    computationalTime = endTime - startTime;

    cerr << " Total Computational Time: ";

    if (computationalTime < 60)
    {
        cerr << computationalTime << " seconds" << endl;
    }
    else
        if ((computationalTime >= 60) && (computationalTime < 60*60))
        {
            cerr << setprecision(3) << computationalTime/60. << " minutes" << endl;
        }
    else
        if (computationalTime >= 60*60)
        {
            cerr << setprecision(3) << computationalTime/(60.*60.) << " hours" << endl;
        }
    else
        if (computationalTime >= 60*60*24)
        {
            cerr << setprecision(3) << computationalTime/(60.*60.*24.) << " days" << endl;
        }

    cerr << "------------------------------------------------" << endl;
}

unsigned int NestedSampler::getNiterations()
{
    return Niterations;
}

unsigned int NestedSampler::getNdimensions()
{
    return Ndimensions;
}

int NestedSampler::getNlivePoints()
{
    return NlivePoints;
}

int NestedSampler::getInitialNlivePoints()
{
    return initialNlivePoints;
}

int NestedSampler::getMinNlivePoints()
{
    return minNlivePoints;
}

double NestedSampler::getLogCumulatedPriorMass()
{
    return logCumulatedPriorMass;
}

double NestedSampler::getLogRemainingPriorMass()
{
    return logRemainingPriorMass;
}

double NestedSampler::getRatioOfRemainderToCurrentEvidence()
{
    return ratioOfRemainderToCurrentEvidence;
}

double NestedSampler::getLogMaxLikelihoodOfLivePoints()
{
    return logMaxLikelihoodOfLivePoints;
}

double NestedSampler::getComputationalTime()
{
    return computationalTime;
}

double NestedSampler::getTerminationFactor()
{
    return terminationFactor;
}

vector<int> NestedSampler::getNlivePointsPerIteration()
{
    return NlivePointsPerIteration;
}

ArrayXXd NestedSampler::getNestedSample()
{
    return nestedSample;
}

ArrayXd NestedSampler::getLogLikelihood()
{
    return logLikelihood;
}

void NestedSampler::setLogEvidence(double newLogEvidence)
{
    logEvidence = newLogEvidence;
}

double NestedSampler::getLogEvidence()
{
    return logEvidence;
}

void NestedSampler::setLogEvidenceError(double newLogEvidenceError)
{
    logEvidenceError = newLogEvidenceError;
}

double NestedSampler::getLogEvidenceError()
{
    return logEvidenceError;
}

void NestedSampler::setInformationGain(double newInformationGain)
{
    informationGain = newInformationGain;
}

double NestedSampler::getInformationGain()
{
    return informationGain;
}

void NestedSampler::setPosteriorSample(ArrayXXd newPosteriorSample)
{
    Ndimensions = newPosteriorSample.rows();
    int Nsamples = newPosteriorSample.cols();
    posteriorSample.resize(Ndimensions, Nsamples);
    posteriorSample = newPosteriorSample;
}

ArrayXXd NestedSampler::getPosteriorSample()
{
    return posteriorSample;
}

void NestedSampler::setLogLikelihoodOfPosteriorSample(ArrayXd newLogLikelihoodOfPosteriorSample)
{
    int Nsamples = newLogLikelihoodOfPosteriorSample.size();
    logLikelihoodOfPosteriorSample.resize(Nsamples);
    logLikelihoodOfPosteriorSample = newLogLikelihoodOfPosteriorSample;
}

ArrayXd NestedSampler::getLogLikelihoodOfPosteriorSample()
{
    return logLikelihoodOfPosteriorSample;
}

void NestedSampler::setLogWeightOfPosteriorSample(ArrayXd newLogWeightOfPosteriorSample)
{
    int Nsamples = newLogWeightOfPosteriorSample.size();
    logWeightOfPosteriorSample.resize(Nsamples);
    logWeightOfPosteriorSample = newLogWeightOfPosteriorSample;
}

ArrayXd NestedSampler::getLogWeightOfPosteriorSample()
{
    return logWeightOfPosteriorSample;
}

void NestedSampler::setOutputPathPrefix(string newOutputPathPrefix)
{
    outputPathPrefix = newOutputPathPrefix;
}

string NestedSampler::getOutputPathPrefix()
{
    return outputPathPrefix;
}

// -------------- ELLIPSOIDAL SAMPLING ALGORITHM --------------------------------------------------
MultiEllipsoidSampler::MultiEllipsoidSampler(const bool printOnTheScreen, vector<Prior*> ptrPriors,
                                             Likelihood &likelihood, Metric &metric, Clusterer &clusterer,
                                             const int initialNlivePoints, const int minNlivePoints,
                                             const double initialEnlargementFraction, const double shrinkingRate)
: NestedSampler(printOnTheScreen, initialNlivePoints, minNlivePoints, ptrPriors, likelihood, metric, clusterer),
  ellipsoidMatrixDecompositionIsSuccessful(true),
  initialEnlargementFraction(initialEnlargementFraction),
  shrinkingRate(shrinkingRate),
  uniform(0.0, 1.0)
{
}

MultiEllipsoidSampler::~MultiEllipsoidSampler()
{

}

bool MultiEllipsoidSampler::drawWithConstraint(const RefArrayXXd totalSample, const unsigned int Nclusters, const vector<int> &clusterIndices,
                                               const vector<int> &clusterSizes, RefArrayXd drawnPoint,
                                               double &logLikelihoodOfDrawnPoint, const int maxNdrawAttempts)
{
    assert(totalSample.cols() == clusterIndices.size());
    assert(drawnPoint.size() == totalSample.rows());
    assert(Nclusters > 0);

    computeEllipsoids(totalSample, Nclusters, clusterIndices, clusterSizes);

    vector<unordered_set<int>> overlappingEllipsoidsIndices;
    findOverlappingEllipsoids(overlappingEllipsoidsIndices);

    if (!ellipsoidMatrixDecompositionIsSuccessful)
    {
        return false;
    }

    vector<double> normalizedHyperVolumes(Nellipsoids);

    for (int n=0; n < Nellipsoids; ++n)
    {
        normalizedHyperVolumes[n] = ellipsoids[n].getHyperVolume();
    }

    double sumOfHyperVolumes = accumulate(normalizedHyperVolumes.begin(), normalizedHyperVolumes.end(), 0.0, plus<double>());

    for (int n=0; n < Nellipsoids; ++n)
    {
        normalizedHyperVolumes[n] /= sumOfHyperVolumes;
    }

    double uniformNumber = uniform(engine);

    double cumulativeHyperVolume = normalizedHyperVolumes[0];
    int indexOfSelectedEllipsoid = 0;

    while (cumulativeHyperVolume < uniformNumber)
    {
        indexOfSelectedEllipsoid++;
        cumulativeHyperVolume += normalizedHyperVolumes[indexOfSelectedEllipsoid];
    }

    bool newPointIsFound = false;

    int NdrawAttempts = 0;

    while ((newPointIsFound == false) & (NdrawAttempts < maxNdrawAttempts))
    {
        NdrawAttempts++;
        ellipsoids[indexOfSelectedEllipsoid].drawPoint(drawnPoint);

        if (!overlappingEllipsoidsIndices[indexOfSelectedEllipsoid].empty())
        {

            int NenclosingEllipsoids = 1;

            for (auto index = overlappingEllipsoidsIndices[indexOfSelectedEllipsoid].begin();
                      index != overlappingEllipsoidsIndices[indexOfSelectedEllipsoid].end();
                      ++index)
            {
                if (ellipsoids[*index].containsPoint(drawnPoint))  NenclosingEllipsoids++;
            }

            uniformNumber = uniform(engine);
            newPointIsFound = (uniformNumber < 1./NenclosingEllipsoids);
            if (!newPointIsFound) continue;
        }
        else
        {
            newPointIsFound = true;
        }

        int beginIndex = 0;

        for (int priorIndex = 0; priorIndex < ptrPriors.size(); ++priorIndex)
        {
            const int NdimensionsOfPrior = ptrPriors[priorIndex]->getNdimensions();

            ArrayXd subsetOfNewPoint = drawnPoint.segment(beginIndex, NdimensionsOfPrior);

            newPointIsFound = ptrPriors[priorIndex]->drawnPointIsAccepted(subsetOfNewPoint);

            if (!newPointIsFound)
            break;

            beginIndex += NdimensionsOfPrior;
        }

        if (!newPointIsFound)
        {

            continue;
        }

        logLikelihoodOfDrawnPoint = likelihood.logValue(drawnPoint);

        if (logLikelihoodOfDrawnPoint < worstLiveLogLikelihood)
        {

            newPointIsFound = false;
        }

    } // end while-loop (newPointIsFound == false)
    if (newPointIsFound)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool MultiEllipsoidSampler::verifySamplerStatus()
{
    if (!ellipsoidMatrixDecompositionIsSuccessful)
    {
        cout << "Ellipsoid Matrix decomposition failed." << endl;
        cout << "Quitting program." << endl;
        return false;
    }
    else
    {
        return true;
    }
}

void MultiEllipsoidSampler::computeEllipsoids(RefArrayXXd const totalSample, const unsigned int Nclusters,
                                              const vector<int> &clusterIndices, const vector<int> &clusterSizes)
{
    assert(totalSample.cols() == clusterIndices.size());
    assert(totalSample.cols() >= Ndimensions + 1);            // At least Ndimensions + 1 points are required to start.

    vector<int> sortedIndices = Functions::argsort(clusterIndices);

    int beginIndex = 0;
    ellipsoids.clear();

    for (int i = 0; i < Nclusters; i++)
    {

        if (clusterSizes[i] < Ndimensions + 1)
        {
            beginIndex += clusterSizes[i];
            continue;
        }
        else
        {
            ArrayXXd sampleOfOneCluster(Ndimensions, clusterSizes[i]);
            for (int n = 0; n < clusterSizes[i]; ++n)
            {
                sampleOfOneCluster.col(n) = totalSample.col(sortedIndices[beginIndex+n]);
            }
            beginIndex += clusterSizes[i];
            double enlargementFraction = updateEnlargementFraction(clusterSizes[i]);

            ellipsoids.push_back(Ellipsoid(sampleOfOneCluster, enlargementFraction));
        }
    }
    Nellipsoids = ellipsoids.size();
}

void MultiEllipsoidSampler::findOverlappingEllipsoids(vector<unordered_set<int>> &overlappingEllipsoidsIndices)
{
    overlappingEllipsoidsIndices.clear();
    overlappingEllipsoidsIndices.resize(ellipsoids.size());

    for (int i = 0; i < Nellipsoids-1; ++i)
    {
        for (int j = i+1; j < Nellipsoids; ++j)
        {
            if (ellipsoids[i].overlapsWith(ellipsoids[j], ellipsoidMatrixDecompositionIsSuccessful))
            {
                overlappingEllipsoidsIndices[i].insert(j);
                overlappingEllipsoidsIndices[j].insert(i);
            }
        }
    }
}

double MultiEllipsoidSampler::updateEnlargementFraction(const int clusterSize)
{
    double updatedEnlargementFraction = initialEnlargementFraction * exp( shrinkingRate * logRemainingPriorMass
                                            + 0.5 * log(static_cast<double>(NlivePoints) / clusterSize) );

    return updatedEnlargementFraction;
}

vector<Ellipsoid> MultiEllipsoidSampler::getEllipsoids()
{
    return ellipsoids;
}

double MultiEllipsoidSampler::getInitialEnlargementFraction()
{
    return initialEnlargementFraction;
}

double MultiEllipsoidSampler::getShrinkingRate()
{
    return shrinkingRate;
}

// -------------- POWER LAW REDUCER ALGORITHM -----------------------------------------------------
PowerlawReducer::PowerlawReducer(NestedSampler &nestedSampler, const double tolerance,
                                       const double exponent, const double terminationFactor)
: LivePointsReducer(nestedSampler),
  tolerance(tolerance),
  exponent(exponent),
  terminationFactor(terminationFactor)
{
    assert(exponent >= 0.0);
    assert(tolerance >= 1.0);
}

PowerlawReducer::~PowerlawReducer()
{
}

int PowerlawReducer::updateNlivePoints()
{
    double ratioOfRemainderToCurrentEvidence = nestedSampler.getRatioOfRemainderToCurrentEvidence();        // Initial prior mass = 1

    NlivePointsAtCurrentIteration = nestedSampler.getNlivePoints();
    double ratio = ratioOfRemainderToCurrentEvidence/terminationFactor;
    int NlivePointsToRemove = pow((tolerance/ratio), exponent);

    updatedNlivePoints = NlivePointsAtCurrentIteration - NlivePointsToRemove;

    if (updatedNlivePoints < nestedSampler.getMinNlivePoints())
    {
        updatedNlivePoints = NlivePointsAtCurrentIteration;
    }
    return updatedNlivePoints;
}

// -------------- FEROZ REDUCER ALGORITHM -----------------------------------------------------
FerozReducer::FerozReducer(NestedSampler &nestedSampler, const double tolerance)
: LivePointsReducer(nestedSampler),
  tolerance(tolerance)
{
}

FerozReducer::~FerozReducer()
{
}

int FerozReducer::updateNlivePoints()
{
    if (nestedSampler.getNiterations() == 0)
    {
        logMaxEvidenceContribution = nestedSampler.getLogMaxLikelihoodOfLivePoints();        // Initial prior mass = 1
    }

    double logMaxEvidenceContributionNew = nestedSampler.getLogLikelihood().maxCoeff() + nestedSampler.getLogRemainingPriorMass();

    NlivePointsAtCurrentIteration = nestedSampler.getNlivePoints();
    double numerator = exp(Functions::logExpDifference(logMaxEvidenceContribution,logMaxEvidenceContributionNew));
    double denominator = exp(logMaxEvidenceContributionNew);
    int NlivePointsToRemove = nestedSampler.getMinNlivePoints() * numerator / (denominator * tolerance);
    updatedNlivePoints = NlivePointsAtCurrentIteration - NlivePointsToRemove;

    if (updatedNlivePoints < nestedSampler.getMinNlivePoints())
    {
        updatedNlivePoints = NlivePointsAtCurrentIteration;
    }

    logMaxEvidenceContribution = logMaxEvidenceContributionNew;
    return updatedNlivePoints;
}

// -------------- RESULT HANDLING PROGRAM ---------------------------------------------------------
Results::Results(NestedSampler &nestedSampler)
: nestedSampler(nestedSampler)
{
}

Results::~Results()
{

}

ArrayXd Results::posteriorProbability()
{
    // Apply Bayes Theorem in logarithmic expression
    ArrayXd logPosteriorDistribution = nestedSampler.getLogWeightOfPosteriorSample() +
                                       nestedSampler.getLogLikelihoodOfPosteriorSample() -
                                       nestedSampler.getLogEvidence();
    ArrayXd posteriorDistribution = logPosteriorDistribution.exp();

    return posteriorDistribution/posteriorDistribution.sum();
}

void Results::writeMarginalDistributionToFile(const int parameterNumber)
{
    int Nrows = parameterValuesInterpolated.size();
    assert(Nrows == marginalDistributionInterpolated.size());

    ArrayXXd parameterDistribution(Nrows,2);
    parameterDistribution.col(0) = parameterValuesInterpolated;
    parameterDistribution.col(1) = marginalDistributionInterpolated;

    ostringstream numberString;
    numberString << setfill('0') << setw(3) << parameterNumber;
    string fileName = "marginalDistribution";
    string fullPath = nestedSampler.getOutputPathPrefix() + fileName + numberString.str() + ".txt";

    ofstream outputFile;
    File::openOutputFile(outputFile, fullPath);
    outputFile << "# Marginal distribution of cubic-spline interpolated points from nested sampling." << endl;
    outputFile << "# Column #1: Parameter values" << endl;
    outputFile << "# Column #2: Marginal distribution values (probability only)" << endl;
    outputFile << scientific << setprecision(9);
    File::arrayXXdToFile(outputFile, parameterDistribution);
    outputFile.close();
}

ArrayXd Results::computeCredibleLimits(const double credibleLevel, const double skewness, const int NinterpolationsPerBin)
{

    int Nbins = parameterValuesRebinned.size();
    int Ninterpolations = Nbins*NinterpolationsPerBin;

    double binWidth = parameterValuesRebinned(1) - parameterValuesRebinned(0);
    double interpolatedBinWidth = binWidth/NinterpolationsPerBin;
    double parameterMinimumRebinned = parameterValuesRebinned.minCoeff();

    parameterValuesInterpolated.resize(Ninterpolations);

    for (int j = 0; j < Ninterpolations; ++j)
    {
        parameterValuesInterpolated(j) = parameterMinimumRebinned + j*interpolatedBinWidth;
    }

    marginalDistributionInterpolated = Functions::cubicSplineInterpolation(parameterValuesRebinned, marginalDistributionRebinned, parameterValuesInterpolated);

    vector<int> selectedIndices = Functions::findArrayIndicesWithinBoundaries(marginalDistributionInterpolated, -1e99, -1e-99);

    for (int i = 0; i < selectedIndices.size(); ++i)
    {
        marginalDistributionInterpolated(selectedIndices[i]) = 0.0;
    }

    marginalDistributionInterpolated /= marginalDistributionInterpolated.sum();

    if (marginalDistributionInterpolated.size() != Ninterpolations)
    {
        Ninterpolations = marginalDistributionInterpolated.size();
        parameterValuesInterpolated.conservativeResize(Ninterpolations);
    }

    int max = 0;
    marginalDistributionMode = marginalDistributionInterpolated.maxCoeff(&max);
    int NbinsLeft = max + 1;
    int NbinsRight = Ninterpolations - NbinsLeft;
    ArrayXd marginalDistributionLeft(NbinsLeft);        // Marginal distribution up to modal value (included)
    ArrayXd parameterValuesLeft(NbinsLeft);             // Parameter range up to modal value (included)
    ArrayXd marginalDistributionRight(NbinsRight);      // Marginal distribution beyond modal value
    ArrayXd parameterValuesRight(NbinsRight);           // Parameter range beyond modal value
    double limitProbabilityRight = marginalDistributionInterpolated(max);
    double limitParameterRight = parameterValuesInterpolated(max);
    double limitProbabilityLeft = limitProbabilityRight;
    double limitParameterLeft = limitParameterRight;

    ArrayXd marginalDifferenceLeft = ArrayXd::Zero(NbinsLeft);
    ArrayXd marginalDifferenceRight = ArrayXd::Zero(NbinsRight);

    marginalDistributionLeft = marginalDistributionInterpolated.segment(0, NbinsLeft);
    marginalDistributionRight = marginalDistributionInterpolated.segment(NbinsLeft, NbinsRight);
    parameterValuesLeft = parameterValuesInterpolated.segment(0, NbinsLeft);
    parameterValuesRight = parameterValuesInterpolated.segment(NbinsLeft, NbinsRight);

    int min = 0;
    double credibleLevelFraction = credibleLevel/100.;
    double totalProbability = 0.0;

    if (skewness >= 0.0)
    {
        int stepRight = 0;

        while (totalProbability < (credibleLevelFraction) && (NbinsLeft + stepRight < Ninterpolations))
        {
            totalProbability = 0.0;

            limitProbabilityRight = marginalDistributionInterpolated(NbinsLeft + stepRight);
            limitParameterRight = parameterValuesInterpolated(NbinsLeft + stepRight);

            marginalDifferenceLeft = (marginalDistributionLeft - limitProbabilityRight).abs();
            limitProbabilityLeft = marginalDifferenceLeft.minCoeff(&min);

            limitProbabilityLeft = marginalDistributionLeft(min);
            limitParameterLeft = parameterValuesLeft(min);

            int intervalSize = NbinsLeft + stepRight - min;
            totalProbability = marginalDistributionInterpolated.segment(min, intervalSize).sum();

            ++stepRight;
        }
    }
    else
    {
        int stepLeft = 0;

        while (totalProbability < (credibleLevelFraction) && (stepLeft < NbinsLeft))
        {
            totalProbability = 0.0;
            limitProbabilityLeft = marginalDistributionInterpolated(NbinsLeft - stepLeft - 1);
            limitParameterLeft = parameterValuesInterpolated(NbinsLeft - stepLeft - 1);

            marginalDifferenceRight = (marginalDistributionRight - limitProbabilityLeft).abs();
            limitProbabilityRight = marginalDifferenceRight.minCoeff(&min);

            limitProbabilityRight = marginalDistributionRight(min);
            limitParameterRight = parameterValuesRight(min);

            int intervalSize = min + stepLeft;
            totalProbability = marginalDistributionInterpolated.segment(NbinsLeft - stepLeft, intervalSize).sum();

            ++stepLeft;
        }
    }

    ArrayXd credibleLimits(2);
    credibleLimits << limitParameterLeft, limitParameterRight;

    return credibleLimits;
}

ArrayXXd Results::parameterEstimation(double credibleLevel, bool writeMarginalDistribution)
{
    int Ndimensions = nestedSampler.getPosteriorSample().rows();
    ArrayXd posteriorDistribution = posteriorProbability();

    int sampleSize = posteriorDistribution.size();
    assert(nestedSampler.getPosteriorSample().cols() == sampleSize);
    ArrayXXd parameterEstimates(Ndimensions, 7);

    parameterValues.resize(sampleSize);
    marginalDistribution.resize(sampleSize);

    for (int i = 0; i < Ndimensions; ++i)
    {

        parameterValues = nestedSampler.getPosteriorSample().row(i);
        marginalDistribution = posteriorDistribution;

        Functions::topDownMergeSort(parameterValues, marginalDistribution);

        double parameterMean = (parameterValues * marginalDistribution).sum();
        parameterEstimates(i,0) = parameterMean;

        double secondMoment = ((parameterValues - parameterMean).pow(2) * marginalDistribution).sum();

        int k = 0;
        double totalProbability = 0.0;
        double parameterMedian = parameterValues(0);

        while (totalProbability < 0.50)
        {
            parameterMedian = parameterValues(k);
            totalProbability += marginalDistribution(k);
            k++;
        }
        parameterEstimates(i,1) = parameterMedian;

        double binWidth = 0;
        double parameterMaximum = parameterValues.maxCoeff();
        double parameterMinimum = parameterValues.minCoeff();

        binWidth = 3.5*sqrt(secondMoment)/pow(sampleSize,1.0/3.0);
        int Nbins = floor((parameterMaximum - parameterMinimum)/binWidth) - 1;

        if (Nbins > 1000)
        {
            Nbins = 1000;
            binWidth = (parameterMaximum - parameterMinimum)/(Nbins*1.0);
        }

        double parameterStart = 0.0;
        double parameterEnd = 0.0;
        parameterValuesRebinned.resize(Nbins);
        parameterValuesRebinned.setZero();
        marginalDistributionRebinned.resize(Nbins);
        marginalDistributionRebinned.setZero();

        int Nshifts = 20;                           // Total number of shifts for the starting point of the rebinning
        double shiftWidth = binWidth/Nshifts;       // Width of the shift bin
        ArrayXd parameterValuesRebinnedPerShift(Nbins);
        parameterValuesRebinnedPerShift.setZero();
        ArrayXd marginalDistributionRebinnedPerShift(Nbins);
        marginalDistributionRebinnedPerShift.setZero();

        for (int k = 0; k < Nshifts; ++k)
        {
            int cumulatedBinSize = 0;
            for (int j = 0; j < Nbins; ++j)
            {

                parameterStart = parameterMinimum + j*binWidth + k*shiftWidth;

                if (j < (Nbins - 1))
                    parameterEnd = parameterMinimum + (j+1)*binWidth + k*shiftWidth;
                else
                    parameterEnd = parameterMaximum;

                int binSize = Functions::countArrayIndicesWithinBoundaries(parameterValues, parameterStart, parameterEnd);
                parameterValuesRebinnedPerShift(j) = (parameterStart + parameterEnd)/2.0;

                if (binSize > 0)
                {
                    marginalDistributionRebinnedPerShift(j) = marginalDistribution.segment(cumulatedBinSize, binSize).sum();
                    cumulatedBinSize += binSize;
                }
                else
                {
                    marginalDistributionRebinnedPerShift(j) = 0.0;
                }
            }

            parameterValuesRebinned += parameterValuesRebinnedPerShift;
            marginalDistributionRebinned += marginalDistributionRebinnedPerShift;
        }

        parameterValuesRebinned /= Nshifts;
        marginalDistributionRebinned /= Nshifts;

        int max = 0;                                    // Subscript corresponding to mode value
        marginalDistributionMode = marginalDistributionRebinned.maxCoeff(&max);
        double parameterMode = parameterValuesRebinned(max);
        parameterEstimates(i,2) = parameterMode;

        parameterEstimates(i,3) = secondMoment;

        double thirdMoment = ((parameterValues - parameterMean).pow(3) * marginalDistribution).sum();
        double skewness = thirdMoment/pow(secondMoment,3.0/2.0);

        ArrayXd credibleLimits(2);
        credibleLimits = computeCredibleLimits(credibleLevel, skewness);

        parameterEstimates(i,4) = credibleLimits(0);
        parameterEstimates(i,5) = credibleLimits(1);

        parameterEstimates(i,6) = skewness;

        if (writeMarginalDistribution)
        {
            writeMarginalDistributionToFile(i);
        }

    }   // END for loop over the parameters
    return parameterEstimates;
}

void Results::writeParametersToFile(string fileName, string outputFileExtension)
{
    string pathPrefix = nestedSampler.getOutputPathPrefix() + fileName;
    ArrayXXd posteriorSample = nestedSampler.getPosteriorSample();
    File::arrayXXdRowsToFiles(posteriorSample, pathPrefix, outputFileExtension);
}

void Results::writeLogLikelihoodToFile(string fileName)
{
    string fullPath = nestedSampler.getOutputPathPrefix() + fileName;

    ofstream outputFile;
    File::openOutputFile(outputFile, fullPath);

    outputFile << "# Posterior sample from nested sampling" << endl;
    outputFile << "# log(Likelihood)" << endl;
    outputFile << scientific << setprecision(9);

    ArrayXd logLikelihoodOfPosteriorSample = nestedSampler.getLogLikelihoodOfPosteriorSample();
    File::arrayXdToFile(outputFile, logLikelihoodOfPosteriorSample);
    outputFile.close();
}

void Results::writeLogWeightsToFile(string fileName)
{
    string fullPath = nestedSampler.getOutputPathPrefix() + fileName;

    ofstream outputFile;
    File::openOutputFile(outputFile, fullPath);

    outputFile << "# Posterior sample from nested sampling" << endl;
    outputFile << "# log(Weight) = log(dX)" << endl;
    outputFile << scientific << setprecision(9);

    ArrayXd logWeightOfPosteriorSample = nestedSampler.getLogWeightOfPosteriorSample();
    File::arrayXdToFile(outputFile, logWeightOfPosteriorSample);
    outputFile.close();
}

void Results::writeEvidenceInformationToFile(string fileName)
{
    string fullPath = nestedSampler.getOutputPathPrefix() + fileName;

    ofstream outputFile;
    File::openOutputFile(outputFile, fullPath);

    outputFile << "# Evidence results from nested sampling" << endl;
    outputFile << scientific << setprecision(9);
    outputFile << "# Skilling's log(Evidence)" << setw(40) << "Skilling's Error log(Evidence)"
    << setw(40) << "Skilling's Information Gain" << endl;
    outputFile << nestedSampler.getLogEvidence() << setw(40) << nestedSampler.getLogEvidenceError()
    << setw(40) << nestedSampler.getInformationGain() << endl;
    outputFile.close();
}

void Results::writePosteriorProbabilityToFile(string fileName)
{
    ArrayXd posteriorDistribution = posteriorProbability();
    string fullPath = nestedSampler.getOutputPathPrefix() + fileName;

    ofstream outputFile;
    File::openOutputFile(outputFile, fullPath);

    outputFile << "# Posterior probability distribution from nested sampling" << endl;
    outputFile << scientific << setprecision(9);
    File::arrayXdToFile(outputFile, posteriorDistribution);
    outputFile.close();
}

void Results::writeParametersSummaryToFile(string fileName, const double credibleLevel, const bool writeMarginalDistribution)
{
    ArrayXXd parameterEstimates = parameterEstimation(credibleLevel, writeMarginalDistribution);

    // Write output ASCII file
    string fullPath = nestedSampler.getOutputPathPrefix() + fileName;
    ofstream outputFile;
    File::openOutputFile(outputFile, fullPath);

    outputFile << "# Summary of Parameter Estimation from nested sampling" << endl;
    outputFile << "# Credible intervals are the shortest credible intervals" << endl;
    outputFile << "# according to the usual definition" << endl;
    outputFile << "# Credible level: " << fixed << setprecision(2) << credibleLevel << " %" << endl;
    outputFile << "# Column #1: I Moment (Mean)" << endl;
    outputFile << "# Column #2: Median" << endl;
    outputFile << "# Column #3: Mode" << endl;
    outputFile << "# Column #4: II Moment (Variance if Normal distribution)" << endl;
    outputFile << "# Column #5: Lower Credible Limit" << endl;
    outputFile << "# Column #6: Upper Credible Limit" << endl;
    outputFile << "# Column #7: Skewness (Asymmetry of the distribution, -1 to the left, +1 to the right, 0 if symmetric)" << endl;
    outputFile << scientific << setprecision(9);
    File::arrayXXdToFile(outputFile, parameterEstimates);
    outputFile.close();
}
// ------------------------------------------------------------------------------------------------
