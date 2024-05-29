package DCGAN.optimizers;

import java.io.Serializable;

public class RMSPropHyperparameters extends OptimizerHyperparameters implements Serializable {
    private static final long serialVersionUID = 1L;
    private double learningRate;
    private double decayRate;
    private double epsilon;

    public RMSPropHyperparameters(double learningRate, double decayRate, double epsilon) {
        super(OptimizerType.RMSPROP);
        this.learningRate = learningRate;
        this.decayRate = decayRate;
        this.epsilon = epsilon;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getDecayRate() {
        return decayRate;
    }

    public void setDecayRate(double decayRate) {
        this.decayRate = decayRate;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
}