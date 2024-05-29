package DCGAN.optimizers;

public enum OptimizerType {


    ADAM("Adam"),
    SGD("SGD"),
    RMSPROP("RMSProp");

    private final String optimizerType;

    OptimizerType(String optimizerType) {
        this.optimizerType = optimizerType;
    }

    public String getOptimizerType() {
        return optimizerType;
    }
}
