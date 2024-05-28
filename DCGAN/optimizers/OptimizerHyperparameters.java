package DCGAN.optimizers;

import java.io.Serializable;

public class OptimizerHyperparameters implements Serializable {
    private static final long serialVersionUID = 1L;
    protected OptimizerType optimizerType;

    public OptimizerHyperparameters(OptimizerType optimizerType) {
        this.optimizerType = optimizerType;
    }

    public OptimizerType getOptimizerType() {
        return optimizerType;
    }

}