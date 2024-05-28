package DCGAN.optimizers;

import java.io.Serializable;

public abstract class Optimizer implements Serializable {
    private static final long serialVersionUID = 1L;

    public static Optimizer createOptimizer(int numParams, OptimizerHyperparameters hyperparameters) {
        return switch (hyperparameters.getOptimizerType()) {
            case ADAM -> new AdamOptimizer(numParams, (AdamHyperparameters) hyperparameters);
            case SGD -> new SGDOptimizer(numParams, (SGDHyperparameters) hyperparameters);
            default ->
                    throw new IllegalArgumentException("Unsupported optimizer type: " + hyperparameters.getOptimizerType());
        };
    }

    public abstract void updateParameters(double[] params, double[] grads);

    public abstract void updateParameters(double[][] params, double[][] grads);

    public abstract void updateParameters(double[][][] params, double[][][] grads);

    public abstract void updateParameters(double[][][][] params, double[][][][] grads);
}