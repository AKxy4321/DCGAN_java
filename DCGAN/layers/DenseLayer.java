package DCGAN.layers;

import DCGAN.optimizers.Optimizer;
import DCGAN.optimizers.OptimizerHyperparameters;
import DCGAN.util.MiscUtils;
import DCGAN.util.ArrayInitializer;

import java.io.Serializable;

public class DenseLayer implements Serializable {
    private static final long serialVersionUID = 1L;

    public double[][] weights;
    public double[] biases;
    double[] input;

    public int outputSize, inputSize;
    Optimizer weightsOptimizer, biasesOptimizer;


    public DenseLayer(int inputSize, int outputSize, OptimizerHyperparameters optimizerHyperparameters) {
        this.outputSize = outputSize;
        this.inputSize = inputSize;

        this.weights = ArrayInitializer.xavierInit2D(inputSize, outputSize);
        this.biases = ArrayInitializer.xavierInit1D(outputSize);

        weightsOptimizer = Optimizer.createOptimizer(inputSize * outputSize, optimizerHyperparameters);
        biasesOptimizer = Optimizer.createOptimizer(outputSize, optimizerHyperparameters);
    }

    public void setOptimizerHyperparameters(OptimizerHyperparameters optimizerHyperparameters) {
        setWeightsOptimizerHyperparameters(optimizerHyperparameters);
        setBiasesOptimizerHyperparameters(optimizerHyperparameters);
    }

    public void setWeightsOptimizerHyperparameters(OptimizerHyperparameters optimizerHyperparameters) {
        weightsOptimizer = Optimizer.createOptimizer(inputSize * outputSize, optimizerHyperparameters);
    }

    public void setBiasesOptimizerHyperparameters(OptimizerHyperparameters optimizerHyperparameters) {
        biasesOptimizer = Optimizer.createOptimizer(outputSize, optimizerHyperparameters);
    }

    public double[] forward(double[] input) {
        this.input = input;
        double[] output = new double[weights[0].length];
        for (int j = 0; j < weights[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < weights.length; i++) {
                sum += input[i] * weights[i][j];
            }
            output[j] = sum + biases[j];
        }
        return output;
    }

    public double[] backward(double[] outputGradient) {
        double[] inputGradient = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double sum = 0;
            for (int j = 0; j < weights[0].length; j++) {
                sum += outputGradient[j] * weights[i][j];
            }
            inputGradient[i] = sum;// + biases[i];
        }

        return inputGradient;
    }


    public double[][] calculateWeightsGradient(double[] outputGradient, double[] input) {
        double[][] weightsGradients = new double[weights.length][weights[0].length];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weightsGradients[i][j] = outputGradient[j] * input[i];
            }
        }
        return weightsGradients;
    }

    public double[] calculateBiasGradient(double[] outputGradient) {
        double[] biasesGradient = new double[biases.length];
        for (int j = 0; j < biases.length; j++) {
            biasesGradient[j] = outputGradient[j];
        }
        return biasesGradient;
    }

    public void updateParameters(double[] outputGradient, double[] input) {
        double[][] weightsGradients = calculateWeightsGradient(outputGradient, input);
        double[] biasesGradients = calculateBiasGradient(outputGradient);

        weightsOptimizer.updateParameters(weights, weightsGradients);
        biasesOptimizer.updateParameters(biases, biasesGradients);
    }

    public void updateParametersBatch(double[][] outputGradients, double[][] inputs) {
        /** from a batch of inputs, for a batch of output gradients, we want to update based on the mean of the weights gradients and the mean of the biases gradients*/

        double[][][] weightsGradients = new double[outputGradients.length][weights.length][weights[0].length];
        double[][] biasesGradients = new double[outputGradients.length][biases.length];
        for (int sample_idx = 0; sample_idx < outputGradients.length; sample_idx++) {
            weightsGradients[sample_idx] = calculateWeightsGradient(outputGradients[sample_idx], inputs[sample_idx]);
            biasesGradients[sample_idx] = calculateBiasGradient(outputGradients[sample_idx]);
        }

        weightsOptimizer.updateParameters(weights, MiscUtils.mean_1st_layer(weightsGradients));
        biasesOptimizer.updateParameters(biases, MiscUtils.mean_1st_layer(biasesGradients));
    }

    @Deprecated
    public void updateParameters(double[] outputGradient) {
        updateParameters(outputGradient, this.input);
    }

    public double[][][] backward2_and_update(double[][][] grad_wrt_input, double[][] grad_wrt_output, double[][] inputs) {
        // TODO: implement
        return new double[0][][];
    }
}