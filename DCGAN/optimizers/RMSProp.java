package DCGAN.optimizers;

import java.io.Serializable;

public class RMSProp extends Optimizer implements Serializable {
    private static final long serialVersionUID = 1L;

    private final RMSPropHyperparameters hyperparameters;
    private double[] cache;
    private int numParams;

    public RMSProp(int numParams, RMSPropHyperparameters hyperparameters) {
        this.numParams = numParams;
        this.hyperparameters = hyperparameters;
        this.cache = new double[numParams];
    }

    public void updateParameters(double[] params, double[] grads) {
        for (int i = 0; i < params.length; i++) {
            cache[i] = hyperparameters.getDecayRate() * cache[i] + (1 - hyperparameters.getDecayRate()) * Math.pow(grads[i], 2);
            params[i] -= hyperparameters.getLearningRate() * grads[i] / (Math.sqrt(cache[i]) + hyperparameters.getEpsilon());
        }
    }

    public void updateParameters(double[][] params, double[][] grads) {
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                flattened_idx = i * params[i].length + j;
                cache[flattened_idx] = hyperparameters.getDecayRate() * cache[flattened_idx] + (1 - hyperparameters.getDecayRate()) * Math.pow(grads[i][j], 2);
                params[i][j] -= hyperparameters.getLearningRate() * grads[i][j] / (Math.sqrt(cache[flattened_idx]) + hyperparameters.getEpsilon());
            }
        }
    }

    public void updateParameters(double[][][] params, double[][][] grads) {
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    flattened_idx = i * params[i].length * params[i][j].length + j * params[i][j].length + k;
                    cache[flattened_idx] = hyperparameters.getDecayRate() * cache[flattened_idx] + (1 - hyperparameters.getDecayRate()) * Math.pow(grads[i][j][k], 2);
                    params[i][j][k] -= hyperparameters.getLearningRate() * grads[i][j][k] / (Math.sqrt(cache[flattened_idx]) + hyperparameters.getEpsilon());
                }
            }
        }
    }

    public void updateParameters(double[][][][] params, double[][][][] grads) {
        for (int i = 0, flattened_idx; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    for (int l = 0; l < params[i][j][k].length; l++) {
                        flattened_idx = i * params[i].length * params[i][j].length * params[i][j][k].length + j * params[i][j].length * params[i][j][k].length + k * params[i][j][k].length + l;
                        cache[flattened_idx] = hyperparameters.getDecayRate() * cache[flattened_idx] + (1 - hyperparameters.getDecayRate()) * Math.pow(grads[i][j][k][l], 2);
                        params[i][j][k][l] -= hyperparameters.getLearningRate() * grads[i][j][k][l] / (Math.sqrt(cache[flattened_idx]) + hyperparameters.getEpsilon());
                    }
                }
            }
        }
    }

}