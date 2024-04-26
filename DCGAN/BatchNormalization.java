package DCGAN;

public class BatchNormalization {
    private final double epsilon = 1e-5;
    private final double momentum = 0.9;
    private double runningMean;
    private double runningVar;
    private double[] gamma;
    private double[] beta;
    private int batchSize;
    private double[] x;
    private double[] xNormalized;

    public BatchNormalization(int inputDim) {
        initializeParameters(inputDim);
    }

    public double[] forward(double[] x, boolean training) {
        this.x = x;
        int inputDim = x.length;
        double[] out = new double[inputDim];

        if (training) {
            batchSize = x.length; // Update batch size
            double mean = calculateMean(x);
            double var = calculateVariance(x, mean);
            double[] xCentered = calculateXCentered(x, mean);
            double[] xNormalized = calculateXNormalized(xCentered, var);
            this.xNormalized = xNormalized;
            for (int i = 0; i < inputDim; i++) {
                out[i] = gamma[i] * xNormalized[i] + beta[i];
            }
            runningMean = updateRunningMean(mean);
            runningVar = updateRunningVar(var);
        } else {
            double[] xNormalized = calculateXNormalized(x, runningMean, runningVar);
            this.xNormalized = xNormalized;
            for (int i = 0; i < inputDim; i++) {
                out[i] = gamma[i] * xNormalized[i] + beta[i];
            }
        }

        return out;
    }

    public double[] backward(double[] dout) {
        double[] dx = new double[dout.length];
        double[] dxNormalized = new double[dout.length];
        double dVar = 0;
        double dMean = 0;
        double invBatchSize = 1.0 / batchSize; // Calculate 1/batchSize

        for (int i = 0; i < dout.length; i++) {
            dVar += dout[i] * (gamma[i] * (1 - epsilon) * Math.pow(runningVar + epsilon, -1.5));
            dMean += dout[i] * (gamma[i] / Math.sqrt(runningVar + epsilon));
            dxNormalized[i] = dout[i] * gamma[i] / Math.sqrt(runningVar + epsilon);
        }

        for (int i = 0; i < dout.length; i++) {
            dx[i] = dxNormalized[i] - (x[i] - runningMean) * dVar * invBatchSize - dMean * invBatchSize;
        }

        return dx;
    }

    public void updateParameters(double[] dout, double learning_rate){
        double[] dGamma = new double[gamma.length];
        double[] dBeta = new double[beta.length];
        double invBatchSize = 1.0 / batchSize; // Calculate 1/batchSize

        for (int i = 0; i < dout.length; i++) {
            dGamma[i] += dout[i] * xNormalized[i];
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

    private double[] calculateXNormalized(double[] xCentered, double var) {
        double[] xNormalized = new double[xCentered.length];
        for (int i = 0; i < xCentered.length; i++) {
            xNormalized[i] = xCentered[i] / Math.sqrt(var + epsilon);
        }
        return xNormalized;
    }

    private double[] calculateXNormalized(double[] x, double runningMean, double runningVar) {
        double[] xNormalized = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            xNormalized[i] = (x[i] - runningMean) / Math.sqrt(runningVar + epsilon);
        }
        return xNormalized;
    }

    private double updateRunningMean(double mean) {
        runningMean = momentum * runningMean + (1 - momentum) * mean;
        return runningMean;
    }

    private double updateRunningVar(double var) {
        runningVar = momentum * runningVar + (1 - momentum) * var;
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
}