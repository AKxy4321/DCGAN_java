package DCGAN;

import java.util.Arrays;

public class BatchNormalization {
    private final double epsilon = 1e-5;
    private final double momentum = 0.9;
    private double[] runningMean;
    private double[] runningVar;
    private double[] gamma;
    private double[] beta;
    private double[] xNormalized;
    private double[] xCentered;
    private double[] mean;
    private double[] var;
    private int batchSize;

    public BatchNormalization() {
    }

    public double[][] forward(double[][] x, boolean training) {
        int inputDim = x[0].length;
        if (gamma == null) {
            initializeParameters(inputDim);
        }

        double[][] out = new double[x.length][inputDim];

        if (training) {
            batchSize = x.length;
            mean = calculateMean(x);
            var = calculateVariance(x);
            xCentered = calculateXCentered(x, mean);
            xNormalized = calculateXNormalized(xCentered, var);
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < inputDim; j++) {
                    out[i][j] = gamma[j] * xNormalized[i * inputDim + j] + beta[j];
                }
            }
            runningMean = updateRunningMean(mean);
            runningVar = updateRunningVar(var);
        } else {
            xNormalized = calculateXNormalized(x, runningMean, runningVar);
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < inputDim; j++) {
                    out[i][j] = gamma[j] * xNormalized[i * inputDim + j] + beta[j];
                }
            }
        }

        return out;
    }

    public double[][] backward(double[][] dout) {
        double[] dGamma = new double[gamma.length];
        double[] dBeta = new double[beta.length];
        double[] dxNormalized = new double[xNormalized.length];
        double[] dVar = new double[var.length];
        double[] dMean = new double[mean.length];
        double[] dx = new double[xCentered.length];
        double[][] dxResult = new double[dx.length / mean.length][mean.length];

        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < gamma.length; j++) {
                dGamma[j] += dout[i][j] * xNormalized[i * gamma.length + j];
                dBeta[j] += dout[i][j];
                dxNormalized[i * gamma.length + j] = dout[i][j] * gamma[j];
            }
        }

        for (int i = 0; i < xCentered.length / mean.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                for (int k = 0; k < gamma.length; k++) {
                    dVar[j] += dxNormalized[i * mean.length + j] * xCentered[i * mean.length + j] * -0.5 * Math.pow(var[j] + epsilon, -1.5);
                    dMean[j] += dxNormalized[i * mean.length + j] * -1 / Math.sqrt(var[j] + epsilon);
                }
            }
        }

        for (int i = 0; i < xCentered.length / mean.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                for (int k = 0; k < gamma.length; k++) {
                    dx[i * mean.length + j] += dxNormalized[i * mean.length + j] / Math.sqrt(var[j] + epsilon) +
                            dVar[j] * 2 * xCentered[i * mean.length + j] / batchSize +
                            dMean[j] / batchSize;
                }
            }
        }

        for (int i = 0; i < gamma.length; i++) {
            gamma[i] -= dGamma[i];
            beta[i] -= dBeta[i];
        }

        for (int i = 0; i < dxResult.length; i++) {
            dxResult[i] = Arrays.copyOfRange(dx, i * mean.length, (i + 1) * mean.length);
        }

        return dxResult;
    }

    private double[] calculateMean(double[][] x) {
        double[] sum = new double[x[0].length];
        for (double[] xi : x) {
            for (int j = 0; j < xi.length; j++) {
                sum[j] += xi[j];
            }
        }
        for (int i = 0; i < sum.length; i++) {
            sum[i] /= x.length;
        }
        return sum;
    }

    private double[] calculateVariance(double[][] x) {
        double[] sumSquare = new double[x[0].length];
        double[] sum = new double[x[0].length];
        for (double[] xi : x) {
            for (int j = 0; j < xi.length; j++) {
                sum[j] += xi[j];
                sumSquare[j] += xi[j] * xi[j];
            }
        }
        double[] variance = new double[x[0].length];
        for (int i = 0; i < variance.length; i++) {
            variance[i] = (sumSquare[i] / x.length) - (sum[i] / x.length) * (sum[i] / x.length);
        }
        return variance;
    }

    private double[] calculateXCentered(double[][] x, double[] mean) {
        double[] xCentered = new double[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                xCentered[i * x[0].length + j] = x[i][j] - mean[j];
            }
        }
        return xCentered;
    }

    private double[] calculateXNormalized(double[] xCentered, double[] var) {
        double[] xNormalized = new double[xCentered.length];
        for (int i = 0; i < xCentered.length; i++) {
            xNormalized[i] = xCentered[i] / Math.sqrt(var[i % var.length] + epsilon);
        }
        return xNormalized;
    }

    private double[] calculateXNormalized(double[][] x, double[] runningMean, double[] runningVar) {
        double[] xNormalized = new double[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                xNormalized[i * x[0].length + j] = (x[i][j] - runningMean[j]) / Math.sqrt(runningVar[j] + epsilon);
            }
        }
        return xNormalized;
    }

    private double[] updateRunningMean(double[] mean) {
        if (runningMean == null) {
            runningMean = new double[mean.length];
        }
        for (int i = 0; i < mean.length; i++) {
            runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
        }
        return runningMean;
    }

    private double[] updateRunningVar(double[] var) {
        if (runningVar == null) {
            runningVar = new double[var.length];
        }
        for (int i = 0; i < var.length; i++) {
            runningVar[i] = momentum * runningVar[i] + (1 - momentum) * var[i];
        }
        return runningVar;
    }

    private void initializeParameters(int inputDim) {
        gamma = new double[inputDim];
        beta = new double[inputDim];
        runningMean = new double[inputDim];
        runningVar = new double[inputDim];
        for (int i = 0; i < inputDim; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    public static void main(String[] args) {
        // Demo input
        double[][] input = {
                {100000.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, -900000.0}
        };

        // Create an instance of BatchNormalization
        BatchNormalization batchNormalization = new BatchNormalization();

        // Forward pass
        double[][] normalizedOutput = batchNormalization.forward(input, true);
        System.out.println("Normalized Output:");
        printMatrix(normalizedOutput);

        // Backward pass (Assuming some gradient values as demo)
        double[][] gradients = {
                {0.1, 0.2, 0.3},
                {0.4, 0.5, 0.6},
                {0.7, 0.8, 0.9}
        };
        double[][] dx = batchNormalization.backward(gradients);
        System.out.println("\nBackward Output (dx):");
        printMatrix(dx);
    }

    // Utility function to print a matrix
    private static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
    }
}
