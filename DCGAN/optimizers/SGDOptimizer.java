package DCGAN.optimizers;

import java.io.Serializable;

public class SGDOptimizer extends Optimizer implements Serializable {
    private static final long serialVersionUID = 1L;
    private final SGDHyperparameters hyperparameters;
    private int numParams;

    public SGDOptimizer(int numParams) {
        this(numParams, new SGDHyperparameters(0.01));
    }

    public SGDOptimizer(int numParams, SGDHyperparameters hyperparameters) {
        this.numParams = numParams;
        this.hyperparameters = hyperparameters;
    }

    public void updateParameters(double[] params, double[] grads) {
        for (int i = 0; i < params.length; i++) {
            params[i] -= hyperparameters.getLearningRate() * grads[i];
        }
    }

    public void updateParameters(double[][] params, double[][] grads) {
        for (int i = 0; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                params[i][j] -= hyperparameters.getLearningRate() * grads[i][j];
            }
        }
    }


    public void updateParameters(double[][][] params, double[][][] grads) {
        for (int i = 0; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    params[i][j][k] -= hyperparameters.getLearningRate() * grads[i][j][k];
                }
            }
        }
    }


    public void updateParameters(double[][][][] params, double[][][][] grads) {
        for (int i = 0; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                for (int k = 0; k < params[i][j].length; k++) {
                    for (int l = 0; l < params[i][j][k].length; l++) {
                        params[i][j][k][l] -= hyperparameters.getLearningRate() * grads[i][j][k][l];
                    }
                }
            }
        }
    }
}