package DCGAN.layers;

import javax.naming.InitialContext;

public class InstanceNormalisation {
    private static final double epsilon = 1e-5;
    double[] mean;
    double[] variance;
    double[][] x;

    public InstanceNormalisation(int batchSize, int inputSize) {
        this.mean = new double[inputSize];
        this.variance = new double[inputSize];
        this.x = new double[batchSize][inputSize];
    }

    public void mean(double[] x, int i) {
        double sum = 0;
        for (double v : x) {
            sum += v;
        }
        this.mean[i] = sum / x.length;
    }

    public void variance(double[] x, int i) {
        double v_sum = 0;
        for (double v : x) {
            v_sum += Math.pow(v - mean[i], 2);
        }
        this.variance[i] = v_sum / x.length;
    }

    public double[] forward(double[] x) {
        mean(x, 0);
        variance(x, 0);
        for(int j = 0 ; j < x.length ; j++) {
            x[j] = (x[j] - mean[0]) / Math.sqrt(variance[0] + epsilon);
        }
        return x;
    }

    public double[][] forwardBatch(double[][] x) {
        this.x = x;
        for(int i = 0 ; i < this.x.length ; i++) {
            mean(x[i], i);
            variance(x[i], i);
            for(int j = 0 ; j < this.x[0].length ; j++) {
                this.x[i][j] = (this.x[i][j] - mean[i]) / Math.sqrt(variance[i] + epsilon);
            }
        }
        return x;
    }

}