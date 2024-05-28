package DCGAN.optimizers;

public enum OptimizerType {


    ADAM("Adam"),
    SGD("SGD");

    private final String optimizerType;

    OptimizerType(String optimizerType) {
        this.optimizerType = optimizerType;
    }

    public String getOptimizerType() {
        return optimizerType;
    }
}
