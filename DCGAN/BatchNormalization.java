package DCGAN;

public class BatchNormalization {
    private final double epsilon = 1e-5;
    private final double momentum = 0.9;
    private double[] runningMean;
    private double[] runningVar;
    private double[] gamma;
    private double[] beta;
    private int batchSize;
    private double[][] x;
    private double[] xNormalized;
    private int inputDim;

    public BatchNormalization(int inputDim) {
        initializeParameters(inputDim);
    }

    public double[][] forwardBatch(double[][] x) {
        this.x = x;
        // forward pass for training on a batch of 1d inputs
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
                xCentered[j][i] += x[j][i] - mean[i];
                var[i] += (x[j][i] - mean[i]) * (x[j][i] - mean[i]);
            }
            var[i] /= batchSize;
        }

        for (int j = 0; j < batchSize; j++)
            xNormalized[j] = calculateXNormalized(xCentered[j], var);

        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < batchSize; j++) {
                out[j][i] = gamma[i] * xNormalized[j][i] + beta[i];
            }
        }

        runningMean = updateRunningMean(mean);
        runningVar = updateRunningVar(var);

        //updating the xNormalized to be used in the backward pass of a batch
        this.xNormalized = UTIL.mean_1st_layer(xNormalized);

        return out;
    }

    public double[] getOutput(double[] x) {
        // normal forward pass without training on a batch
        int inputDim = x.length;
        this.inputDim = inputDim;
        double[] out = new double[inputDim];

        double[] xNormalized = calculateXNormalized(x, runningMean, runningVar);
        this.xNormalized = xNormalized;
        for (int i = 0; i < inputDim; i++) {
            out[i] = gamma[i] * xNormalized[i] + beta[i];
        }

        return out;
    }

    public double[][] backward(double[][] dout) {
        this.inputDim = dout[0].length;
        double[][] dx = new double[batchSize][inputDim];
        double[][] dxNormalized = new double[batchSize][inputDim];
        double[] dVar = new double[inputDim];
        double[] dMean = new double[inputDim];
        double invBatchSize = 1.0 / batchSize; // Calculate 1/batchSize

        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dVar[j] += dout[i][j] * (gamma[j] * (1 - epsilon) * Math.pow(runningVar[j] + epsilon, -1.5));
                dMean[j] += dout[i][j] * (gamma[j] / Math.sqrt(runningVar[j] + epsilon));
                dxNormalized[i][j] = dout[i][j] * gamma[j] / Math.sqrt(runningVar[j] + epsilon);
            }
        }
        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dx[i][j] = dxNormalized[i][j] - (x[i][j] - runningMean[j]) * dVar[j] * invBatchSize - dMean[j] * invBatchSize;
            }
        }
        return dx;
    }

    public void updateParameters(double[] dout, double learning_rate) {
        double[] dGamma = new double[gamma.length];
        double[] dBeta = new double[beta.length];
        double invBatchSize = 1.0 / batchSize; // Calculate 1/batchSize

        for (int i = 0; i < dout.length; i++) {
            dGamma[i] += dout[i] * xNormalized[i]; // dout[i] *
            dBeta[i] += dout[i];
        }

        for (int i = 0; i < gamma.length; i++) {
            gamma[i] -= learning_rate * dGamma[i] * invBatchSize;
            beta[i] -= learning_rate * dBeta[i] * invBatchSize;
        }
    }

    private double calculateMean(double[] x) {
        double sum = 0;
        for (double xi : x) {
            sum += xi;
        }
        return sum / x.length;
    }

    private double calculateVariance(double[] x, double mean) {
        double sumSquare = 0;
        for (double xi : x) {
            sumSquare += (xi - mean) * (xi - mean);
        }
        return sumSquare / (x.length - 1); // Unbiased variance
    }

    private double[] calculateXCentered(double[] x, double mean) {
        double[] xCentered = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            xCentered[i] = x[i] - mean;
        }
        return xCentered;
    }


    private double[] calculateXNormalized(double[] xCentered, double[] var) {
        double[] xNormalized = new double[xCentered.length];
        for (int i = 0; i < xCentered.length; i++) {
            xNormalized[i] = xCentered[i] / Math.sqrt(var[i] + epsilon);
        }
        return xNormalized;
    }

    private double[] calculateXNormalized(double[] x, double[] runningMean, double[] runningVar) {
        double[] xNormalized = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            xNormalized[i] = (x[i] - runningMean[i]) / Math.sqrt(runningVar[i] + epsilon);
        }
        return xNormalized;
    }

    private double[] updateRunningMean(double[] mean) {
        double runningMean[] = new double[mean.length];
        for (int i = 0; i < runningMean.length; i++)
            runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
        return runningMean;
    }

    private double[] updateRunningVar(double[] var) {
        double runningVar[] = new double[var.length];
        for (int i = 0; i < runningVar.length; i++)
            runningVar[i] = momentum * runningVar[i] + (1 - momentum) * var[i];
        return runningVar;
    }

    private void initializeParameters(int inputDim) {
        gamma = new double[inputDim];
        beta = new double[inputDim];
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
        batch1.updateParameters(new double[]{1, 2, 3}, 0.01);

        double[] x1 = {2, 4, 6};
        double[] out1 = batch1.getOutput(x1);
        for (double col : out1) {
            System.out.print(col + " ");
        }
    }
}