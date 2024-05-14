package DCGAN.layers;

import DCGAN.UTIL;

import java.util.Arrays;

public class BatchNormalization {
    private final double epsilon = 1e-5;
    //    private final double momentum = 0;
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

        for (int feature_idx = 0; feature_idx < inputDim; feature_idx++) {
            for (int sample_idx = 0; sample_idx < batchSize; sample_idx++) {
                mean[feature_idx] += x[sample_idx][feature_idx];
            }
            mean[feature_idx] /= batchSize;
        }

        for (int feature_idx = 0; feature_idx < inputDim; feature_idx++) {
            for (int sample_idx = 0; sample_idx < batchSize; sample_idx++) {
                xCentered[sample_idx][feature_idx] = x[sample_idx][feature_idx] - mean[feature_idx];
                var[feature_idx] += xCentered[sample_idx][feature_idx] * xCentered[sample_idx][feature_idx];
            }
            var[feature_idx] /= batchSize;
        }

        for (int sample_idx = 0; sample_idx < batchSize; sample_idx++) {
            for (int feature_idx = 0; feature_idx < inputDim; feature_idx++) {
                xNormalized[sample_idx][feature_idx] = xCentered[sample_idx][feature_idx] / Math.sqrt(var[feature_idx] + epsilon);

                out[sample_idx][feature_idx] = gamma[feature_idx] * xNormalized[sample_idx][feature_idx] + beta[feature_idx];

                if (Double.isNaN(out[sample_idx][feature_idx])) {
                    System.err.println("NaN value found");
                }
            }
        }

        for (int feature_idx = 0; feature_idx < inputDim; feature_idx++) {
            runningMean[feature_idx] = mean[feature_idx];
            runningVar[feature_idx] = var[feature_idx];
        }
        return out;
    }

    public double[] getOutput(double[] x) {
        int inputDim = x.length;
        this.inputDim = inputDim;
        double[] out = new double[inputDim];

        double[] xNormalized = calculateXNormalized(x, runningMean, runningVar);

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


        for (int sample_idx = 0; sample_idx < dout.length; sample_idx++) {
            for (int feature_idx = 0; feature_idx < dout[0].length; feature_idx++) {
                dxNormalized[sample_idx][feature_idx] = dout[sample_idx][feature_idx] * gamma[feature_idx];

                dVar[feature_idx] += dxNormalized[sample_idx][feature_idx] * (x[sample_idx][feature_idx] - runningMean[feature_idx])
                        * (-0.5) * Math.pow(runningVar[feature_idx] + epsilon, -1.5);
            }
        }

        for (int sample_idx = 0; sample_idx < dout.length; sample_idx++) {
            for (int feature_idx = 0; feature_idx < dout[0].length; feature_idx++) {
                dMean[feature_idx] += -1 * (dxNormalized[sample_idx][feature_idx] / (Math.sqrt(runningVar[feature_idx] + epsilon)));
            }
        }

        for (int sample_idx = 0; sample_idx < dout.length; sample_idx++) {
            for (int feature_idx = 0; feature_idx < dout[0].length; feature_idx++) {
                dx[sample_idx][feature_idx] = dxNormalized[sample_idx][feature_idx] / Math.sqrt(runningVar[feature_idx] + epsilon)
                        + dVar[feature_idx] * 2 * (x[sample_idx][feature_idx] - runningMean[feature_idx]) / batchSize
                        + dMean[feature_idx] / batchSize;
            }
        }

        return dx;
    }

    public void updateParameters(double[][] dout, double learning_rate) {
        /**
         * Parameters are updated separately in updateParameters function. The input is calculated in the backward function
         * */

        double[] dGamma = new double[inputDim];
        double[] dBeta = new double[inputDim];

        // Compute dGamma and dBeta
        for (int feature_idx = 0; feature_idx < inputDim; feature_idx++) {
            for (int sample_idx = 0; sample_idx < batchSize; sample_idx++) {
                dGamma[feature_idx] += dout[sample_idx][feature_idx] * xNormalized[sample_idx][feature_idx]; // Gradient of loss w.r.t. gamma
                dBeta[feature_idx] += dout[sample_idx][feature_idx]; // Gradient of loss w.r.t. beta
            }
        }

        for (int feature_idx = 0; feature_idx < gamma.length; feature_idx++) {
            gamma[feature_idx] -= learning_rate * dGamma[feature_idx];
            beta[feature_idx] -= learning_rate * dBeta[feature_idx];

            // You shouldn't update the variance and mean during backward pass. It is not supposed to be learnable
//            runningMean[i] -= learning_rate * dMean[i];
//            runningVar[i] -= learning_rate * dVar[i];
        }
    }

    private double[] calculateXNormalized(double[] x, double[] mean, double[] var) {
        double[] xNormalized = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            xNormalized[i] = (x[i] - mean[i]) / Math.sqrt(var[i] + epsilon);
        }
        return xNormalized;
    }

//    private void updateRunningMean(double[] mean) {
//        for (int i = 0; i < runningMean.length; i++)
//            runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
//    }
//    // new_coeff = prev_coeff * momentum + (1 - momentum) * new_coeff
//
//    private void updateRunningVar(double[] var) {
//        for (int i = 0; i < runningVar.length; i++)
//            runningVar[i] = Math.max(momentum * runningVar[i] + (1 - momentum) * var[i], epsilon);
//    }
}