package DCGAN.optimizers;

public class AdamOptimizer {
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private double[] m;
    private double[] v;
    private int t;
    private int numParams;

    public AdamOptimizer(int numParams, double learningRate, double beta1, double beta2, double epsilon) {
        this.numParams = numParams;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.m = new double[numParams];
        this.v = new double[numParams];
        this.t = 0;
    }

    public void updateParameters(double[] params, double[] grads) {
        t++;
        for (int i = 0; i < params.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1 - beta2) * Math.pow(grads[i], 2);
            double mHat = m[i] / (1 - Math.pow(beta1, t));
            double vHat = v[i] / (1 - Math.pow(beta2, t));
            params[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }

    public void updateParameters(double[][] params, double[][] grads) {
        t++;
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                flattened_idx = i * params[i].length + j;
                m[flattened_idx] = beta1 * m[flattened_idx] + (1 - beta1) * grads[i][j];
                v[flattened_idx] = beta2 * v[flattened_idx] + (1 - beta2) * Math.pow(grads[i][j], 2);
                double mHat = m[flattened_idx] / (1 - Math.pow(beta1, t));
                double vHat = v[flattened_idx] / (1 - Math.pow(beta2, t));
                params[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }

    public void updateParameters(double[][][] params, double[][][] grads) {
        t++;
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    flattened_idx = i * params[i].length * params[i][j].length + j * params[i][j].length + k;
                    m[flattened_idx] = beta1 * m[flattened_idx] + (1 - beta1) * grads[i][j][k];
                    v[flattened_idx] = beta2 * v[flattened_idx] + (1 - beta2) * Math.pow(grads[i][j][k], 2);
                    double mHat = m[flattened_idx] / (1 - Math.pow(beta1, t));
                    double vHat = v[flattened_idx] / (1 - Math.pow(beta2, t));
                    params[i][j][k] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }
        }
    }

    public void updateParameters(double[][][][] params, double[][][][] grads){
        t++;
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    for (int l = 0; l < params[i][j][k].length; l++) {
                        flattened_idx = i * params[i].length * params[i][j].length * params[i][j][k].length + j * params[i][j].length * params[i][j][k].length + k * params[i][j][k].length + l;
                        m[flattened_idx] = beta1 * m[flattened_idx] + (1 - beta1) * grads[i][j][k][l];
                        v[flattened_idx] = beta2 * v[flattened_idx] + (1 - beta2) * Math.pow(grads[i][j][k][l], 2);
                        double mHat = m[flattened_idx] / (1 - Math.pow(beta1, t));
                        double vHat = v[flattened_idx] / (1 - Math.pow(beta2, t));
                        params[i][j][k][l] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                    }
                }
            }
        }
    }
}