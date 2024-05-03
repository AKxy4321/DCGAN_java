package DCGAN.layers;

import DCGAN.UTIL;

import java.util.Arrays;

import java.util.Arrays;

public class BatchNormalization {
    private final double epsilon = 1e-5;
    private final double momentum = 0.9;
    private double[] runningMean;
    private double[] runningVar;
    private double[] gamma;
    private double[] beta;
    private int batchSize;

    private int inputDim;

    private double[][] x;
    private double[] xNormalized;

    public BatchNormalization(int inputDim) {
        initializeParameters(inputDim);
    }

    public double[][] forwardBatch(double[][] x) {
        this.x = x;
        this.batchSize = x.length;
        int inputDim = x[0].length;
        double[][] out = new double[batchSize][inputDim];

        double[] mean = new double[inputDim];
        double[] var = new double[inputDim];
        double[][] xCentered = new double[batchSize][inputDim];
        double[][] xNormalized = new double[batchSize][inputDim];

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
            xNormalized[j] = calculateXNormalized(xCentered[j], var);
            for (int i = 0; i < inputDim; i++) {
                out[j][i] = gamma[i] * xNormalized[j][i] + beta[i];
            }
        }

        updateRunningMean(mean);
        updateRunningVar(var);

        this.xNormalized = UTIL.mean_1st_layer(xNormalized);

        return out;
    }

    public double[] getOutput(double[] x) {
        int inputDim = x.length;
        this.inputDim = inputDim;
        double[] out = new double[inputDim];

        double[] xNormalized = calculateXNormalized(runningMean, runningVar);
        this.xNormalized = xNormalized;
        for (int i = 0; i < inputDim; i++) {
            out[i] = gamma[i] * xNormalized[i] + beta[i];
        }

        return out;
    }

    public double[][] backward(double[][] dout) {
        double[][] dx = new double[batchSize][inputDim];
        double[][] dxNormalized = new double[batchSize][inputDim];
        double[] dVar = new double[inputDim];
        double[] dMean = new double[inputDim];
        double invBatchSize = 1.0 / batchSize;

        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dVar[j] += dout[i][j] * gamma[j] * (xNormalized[j] - runningMean[j]) * -0.5 * Math.pow(runningVar[j] + epsilon, -1.5);
                dMean[j] += dout[i][j] * gamma[j] / Math.sqrt(runningVar[j] + epsilon);
                dxNormalized[i][j] = dout[i][j] * gamma[j] / Math.sqrt(runningVar[j] + epsilon);
            }
        }

        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dx[i][j] = dxNormalized[i][j] + dVar[j] * 2 * (xNormalized[j] - runningMean[j]) * invBatchSize + dMean[j] * invBatchSize;
            }
        }
        return dx;
    }

    public void updateParameters(double[][] dout, double learning_rate) {
        double[][] dxNormalized = new double[batchSize][inputDim];
        double[] dVar = new double[inputDim];
        double[] dMean = new double[inputDim];
        double invBatchSize = 1.0 / batchSize;

        int m = dout.length; // Number of samples in the batch
        int n = dout[0].length; // Number of features

        double[] dGamma = new double[n];
        double[] dBeta = new double[n];

        // Compute dGamma and dBeta
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                dGamma[j] += dout[i][j] * xNormalized[i]; // Gradient of loss w.r.t. gamma
                dBeta[j] += dout[i][j]; // Gradient of loss w.r.t. beta
            }
        }

        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dVar[j] += dout[i][j] * gamma[j] * (xNormalized[j] - runningMean[j]) * -0.5 * Math.pow(runningVar[j] + epsilon, -1.5);
                dMean[j] += dout[i][j] * gamma[j] / Math.sqrt(runningVar[j] + epsilon);
                dxNormalized[i][j] = dout[i][j] * gamma[j] / Math.sqrt(runningVar[j] + epsilon);
            }
        }

        for (int i = 0; i < gamma.length; i++) {
            gamma[i] -= learning_rate * dGamma[i];
            beta[i] -= learning_rate * dBeta[i];
            runningMean[i] -= learning_rate * dMean[i];
            runningVar[i] -= learning_rate * dVar[i];
        }
    }

    private double[] calculateXNormalized(double[] xCentered, double[] var) {
        double[] xNormalized = new double[xCentered.length];
        for (int i = 0; i < xCentered.length; i++) {
            xNormalized[i] = xCentered[i] / Math.sqrt(var[i] + epsilon);
        }
        return xNormalized;
    }

    private void updateRunningMean(double[] mean) {
        for (int i = 0; i < runningMean.length; i++)
            runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
    }

    private void updateRunningVar(double[] var) {
        for (int i = 0; i < runningVar.length; i++)
            runningVar[i] = momentum * runningVar[i] + (1 - momentum) * var[i];
    }

    private void initializeParameters(int inputDim) {
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

    public static void main(String[] args) {
        BatchNormalization batch1 = new BatchNormalization(3);
        double[][] x = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] out = batch1.forwardBatch(x);
        for (double[] row : out) {
            for (double col : row) {
                System.out.print(col + " ");
            }
            System.out.println();
        }
        double[][] dx = batch1.backward(new double[][]{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
//        batch1.updateParameters(new double[]{1, 1, 1}, new double[]{1, 1, 1}, new double[]{1, 1, 1}, new double[]{1, 1, 1}, 0.01);

        double[] x1 = {2, 4, 6};
        double[] out1 = batch1.getOutput(x1);
        for (double col : out1) {
            System.out.print(col + " ");
        }
    }
}