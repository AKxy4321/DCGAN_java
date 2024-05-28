package DCGAN.optimizers;

import java.io.Serializable;

public class AdamHyperparameters extends OptimizerHyperparameters implements Serializable {
    private static final long serialVersionUID = 1L;
    private double learningRate;
    private double beta1;
    private double beta2;
    private double epsilon;

    public AdamHyperparameters(double learningRate, double beta1, double beta2, double epsilon) {
        super(OptimizerType.ADAM);
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getBeta1() {
        return beta1;
    }

    public void setBeta1(double beta1) {
        this.beta1 = beta1;
    }

    public double getBeta2() {
        return beta2;
    }

    public void setBeta2(double beta2) {
        this.beta2 = beta2;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
}
