package DCGAN;


public class BatchNormalization {
    private final double epsilon = 1e-5;
    private final double momentum = 0.9;
    private double runningMean;
    private double runningVar;
    private double[] gamma;
    private double[] beta;
    private double[] xNormalized;
    private double[] xCentered;
    private double var;
    private int batchSize;

    public BatchNormalization() {
    }

    public double[] forward(double[] x, boolean training) {
        int inputDim = x.length;
        if (gamma == null) {
            initializeParameters(inputDim);
        }

        double[] out = new double[inputDim];

        if (training) {
            batchSize = 1; // Since we're operating on a 1D array, batchSize is 1
            double mean = calculateMean(x);
            var = calculateVariance(x);
            xCentered = calculateXCentered(x, mean);
            xNormalized = calculateXNormalized(xCentered, var);
            for (int i = 0; i < inputDim; i++) {
                out[i] = gamma[i] * xNormalized[i] + beta[i];
            }
            runningMean = updateRunningMean(mean);
            runningVar = updateRunningVar(var);
        } else {
            xNormalized = calculateXNormalized(x, runningMean, runningVar);
            for (int i = 0; i < inputDim; i++) {
                out[i] = gamma[i] * xNormalized[i] + beta[i];
            }
        }

        return out;
    }

    public double[] backward(double[] dout, double learning_rate) {
        double[] dGamma = new double[gamma.length];
        double[] dBeta = new double[beta.length];
        double[] dxNormalized = new double[xNormalized.length];
        double dVar = 0;
        double dMean = 0;
        double[] dx = new double[xCentered.length];

        for (int i = 0; i < dout.length; i++) {
            dGamma[i] += dout[i] * xNormalized[i];
            dBeta[i] += dout[i];
            dxNormalized[i] = dout[i] * gamma[i];
        }

        for (int i = 0; i < xCentered.length; i++) {
            dVar += dxNormalized[i] * xCentered[i] * -0.5 * Math.pow(var + epsilon, -1.5);
            dMean += dxNormalized[i] * -1 / Math.sqrt(var + epsilon);
        }

        for (int i = 0; i < xCentered.length; i++) {
            dx[i] += dxNormalized[i] / Math.sqrt(var + epsilon) +
                    dVar * 2 * xCentered[i] / batchSize +
                    dMean / batchSize;
        }

        for (int i = 0; i < gamma.length; i++) {
            gamma[i] -= learning_rate * dGamma[i];
            beta[i] -= learning_rate * dBeta[i];
        }

        return dx;
    }

    private double calculateMean(double[] x) {
        double sum = 0;
        for (double xi : x) {
            sum += xi;
        }
        return sum / x.length;
    }

    private double calculateVariance(double[] x) {
        double sumSquare = 0;
        double sum = 0;
        for (double xi : x) {
            sum += xi;
            sumSquare += xi * xi;
        }
        return (sumSquare / x.length) - (sum / x.length) * (sum / x.length);
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
