package DCGAN.optimizers;

public class SGDHyperparameters extends OptimizerHyperparameters implements java.io.Serializable{
    private static final long serialVersionUID = 1L;
    private double learningRate;

    public SGDHyperparameters(double learningRate) {
        super(OptimizerType.SGD);
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}