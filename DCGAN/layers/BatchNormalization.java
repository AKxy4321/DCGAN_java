package DCGAN.layers;

import DCGAN.UTIL;

import java.util.Arrays;

public class BatchNormalization {
    private final double epsilon = 1e-5;
    private final double momentum = 0.8;
    private double[] runningMean;
    private double[] runningVar;
    private double[] gamma;
    private double[] beta;
    private int batchSize;

    private int inputDim;

    private double[][] x;

    double[][] xNormalized;

    // backpropagation variables
    double[] dMean;
    double[] dVar;


    public BatchNormalization(int inputDim) {
        gamma = new double[inputDim];
        beta = new double[inputDim];
        runningMean = new double[inputDim];
        runningVar = new double[inputDim];
        this.inputDim = inputDim;
        for (int i = 0; i < inputDim; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    public double[][] forwardBatch(double[][] x) {
        this.x = x;
        this.batchSize = x.length;
        int inputDim = x[0].length;
        double[][] out = new double[batchSize][inputDim];

        double[] mean = new double[inputDim];
        double[] var = new double[inputDim];
        double[][] xCentered = new double[batchSize][inputDim];
        xNormalized = new double[batchSize][inputDim];

        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < batchSize; j++) {
                mean[i] += x[j][i];
            }
            mean[i] /= batchSize;
        }


        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < batchSize; j++) {
                xCentered[j][i] = x[j][i] - mean[i];
                var[i] += xCentered[j][i] * xCentered[j][i];
            }
            var[i] /= batchSize;
        }

        for (int j = 0; j < batchSize; j++) {

            for (int i = 0; i < inputDim; i++)
                xNormalized[j][i] = xCentered[j][i] / Math.sqrt(var[i] + epsilon);

            for (int i = 0; i < inputDim; i++) {
                out[j][i] = gamma[i] * xNormalized[j][i] + beta[i];
            }
        }

        updateRunningMean(mean);
        updateRunningVar(var);

        return out;
    }

    public double[] getOutput(double[] x) {
        int inputDim = x.length;
        this.inputDim = inputDim;
        double[] out = new double[inputDim];

        double[] xNormalized = calculateXNormalized(x, runningMean, runningVar);
//        this.xNormalized = xNormalized;
        for (int i = 0; i < inputDim; i++) {
            out[i] = gamma[i] * xNormalized[i] + beta[i];
        }

        return out;
    }

    public double[][] backward(double[][] dout) {
        /**
         * Calculates the gradient w.r.t the input to this layer and returns it, but it doesn't update any parameters used here.
         * Parameters are updated separately in updateParameters function
         *
         * reference : https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm
         *
         * */

        double[][] dx = new double[batchSize][inputDim];
        double[][] dxNormalized = new double[batchSize][inputDim];
        this.dVar = new double[inputDim];
        this.dMean = new double[inputDim];


        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dxNormalized[i][j] = dout[i][j] * gamma[j];

                dVar[j] += dxNormalized[i][j] * (x[i][j] - runningMean[j])
                        * (-0.5) * Math.pow(runningVar[j] + epsilon, -1.5);
            }
        }

        double[][] xCentered = new double[batchSize][inputDim];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputDim; j++) {
                xCentered[i][j] = x[i][j] - runningMean[j];
            }
        }

        double[] xCenteredSum = new double[inputDim];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputDim; j++) {
                xCenteredSum[j] += xCentered[i][j];
            }
        }

        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dMean[j] += -1 * (dxNormalized[i][j] / (Math.sqrt(runningVar[j] + epsilon)))
                        + dVar[j] * (-2) * xCenteredSum[j] / batchSize;
            }
        }

        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dx[i][j] = dxNormalized[i][j] / Math.sqrt(runningVar[j] + epsilon)
                        + dVar[j] * 2 * (x[i][j] - runningMean[j]) / batchSize
                        + dMean[j] / batchSize;
            }
        }

        return dx;
    }

    public void updateParameters(double[][] dout, double learning_rate) {
        /**
         * Parameters are updated separately in updateParameters function. The input is calculated in the backward function
         * */

        int m = dout.length; // Number of samples in the batch
        int n = dout[0].length; // Number of features

        double[] dGamma = new double[n];
        double[] dBeta = new double[n];

        // Compute dGamma and dBeta
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                dGamma[j] += dout[i][j] * xNormalized[i][j]; // Gradient of loss w.r.t. gamma
                dBeta[j] += dout[i][j]; // Gradient of loss w.r.t. beta
            }
        }

        for (int i = 0; i < gamma.length; i++) {
            gamma[i] -= learning_rate * dGamma[i];
            beta[i] -= learning_rate * dBeta[i];
            runningMean[i] -= learning_rate * dMean[i];
            runningVar[i] -= learning_rate * dVar[i];
        }
    }

    private double[] calculateXNormalized(double[] x, double[] mean, double[] var) {
        double[] xNormalized = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            xNormalized[i] = (x[i] - mean[i]) / Math.sqrt(var[i] + epsilon);
        }
        return xNormalized;
    }

    private void updateRunningMean(double[] mean) {
        for (int i = 0; i < runningMean.length; i++)
            runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
    }
    // new_coeff = prev_coeff * momentum + (1 - momentum) * new_coeff

    private void updateRunningVar(double[] var) {
        for (int i = 0; i < runningVar.length; i++)
            runningVar[i] = Math.max(momentum * runningVar[i] + (1 - momentum) * var[i], epsilon);
    }
}