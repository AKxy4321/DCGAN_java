package DCGAN.optimizers;

import java.io.Serializable;

public class AdamOptimizer extends Optimizer implements Serializable {
    private static final long serialVersionUID = 1L;

    private final AdamHyperparameters hyperparameters;
    private double[] m;
    private double[] v;
    private int t;
    private int numParams;

    // default initialization
    public AdamOptimizer(int numParams) {
        this(numParams, new AdamHyperparameters(1e-3, 0.9, 0.999, 1e-8)); // initializing with default hyperparameters
    }


    public AdamOptimizer(int numParams, AdamHyperparameters hyperparameters) {
        this.numParams = numParams;
        this.hyperparameters = hyperparameters;
        this.m = new double[numParams];
        this.v = new double[numParams];
        this.t = 0;
    }

    public void updateParameters(double[] params, double[] grads) {
        t++;
        for (int i = 0; i < params.length; i++) {
            m[i] = hyperparameters.getBeta1() * m[i] + (1 - hyperparameters.getBeta1()) * grads[i];
            v[i] = hyperparameters.getBeta2() * v[i] + (1 - hyperparameters.getBeta2()) * Math.pow(grads[i], 2);
            double mHat = m[i] / (1 - Math.pow(hyperparameters.getBeta1(), t));
            double vHat = v[i] / (1 - Math.pow(hyperparameters.getBeta2(), t));
            params[i] -= hyperparameters.getLearningRate() * mHat / (Math.sqrt(vHat) + hyperparameters.getEpsilon());
        }
    }

    public void updateParameters(double[][] params, double[][] grads) {
        t++;
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                flattened_idx = i * params[i].length + j;
                m[flattened_idx] = hyperparameters.getBeta1() * m[flattened_idx] + (1 - hyperparameters.getBeta1()) * grads[i][j];
                v[flattened_idx] = hyperparameters.getBeta2() * v[flattened_idx] + (1 - hyperparameters.getBeta2()) * Math.pow(grads[i][j], 2);
                double mHat = m[flattened_idx] / (1 - Math.pow(hyperparameters.getBeta1(), t));
                double vHat = v[flattened_idx] / (1 - Math.pow(hyperparameters.getBeta2(), t));
                params[i][j] -= hyperparameters.getLearningRate() * mHat / (Math.sqrt(vHat) + hyperparameters.getEpsilon());
            }
        }
    }

    public void updateParameters(double[][][] params, double[][][] grads) {
        t++;
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    flattened_idx = i * params[i].length * params[i][j].length + j * params[i][j].length + k;
                    m[flattened_idx] = hyperparameters.getBeta1() * m[flattened_idx] + (1 - hyperparameters.getBeta1()) * grads[i][j][k];
                    v[flattened_idx] = hyperparameters.getBeta2() * v[flattened_idx] + (1 - hyperparameters.getBeta2()) * Math.pow(grads[i][j][k], 2);
                    double mHat = m[flattened_idx] / (1 - Math.pow(hyperparameters.getBeta1(), t));
                    double vHat = v[flattened_idx] / (1 - Math.pow(hyperparameters.getBeta2(), t));
                    params[i][j][k] -= hyperparameters.getLearningRate() * mHat / (Math.sqrt(vHat) + hyperparameters.getEpsilon());
                }
            }
        }
    }

    public void updateParameters(double[][][][] params, double[][][][] grads) {
        t++;
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    for (int l = 0; l < params[i][j][k].length; l++) {
                        flattened_idx = i * params[i].length * params[i][j].length * params[i][j][k].length + j * params[i][j].length * params[i][j][k].length + k * params[i][j][k].length + l;
                        m[flattened_idx] = hyperparameters.getBeta1() * m[flattened_idx] + (1 - hyperparameters.getBeta1()) * grads[i][j][k][l];
                        v[flattened_idx] = hyperparameters.getBeta2() * v[flattened_idx] + (1 - hyperparameters.getBeta2()) * Math.pow(grads[i][j][k][l], 2);
                        double mHat = m[flattened_idx] / (1 - Math.pow(hyperparameters.getBeta1(), t));
                        double vHat = v[flattened_idx] / (1 - Math.pow(hyperparameters.getBeta2(), t));
                        params[i][j][k][l] -= hyperparameters.getLearningRate() * mHat / (Math.sqrt(vHat) + hyperparameters.getEpsilon());
                    }
                }
            }
        }
    }
}