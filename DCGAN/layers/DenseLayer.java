package DCGAN.layers;

import DCGAN.optimizers.AdamOptimizer;
import DCGAN.util.MiscUtils;
import DCGAN.util.XavierInitializer;

public class DenseLayer {
    private double[][] weights;
    private double[] biases;
    double[] input;

    public int outputSize, inputSize;
    AdamOptimizer weightsOptimizer, biasesOptimizer;


    public DenseLayer(int inputSize, int outputSize) {
        this(inputSize, outputSize, 0.001);
    }

    public DenseLayer(int inputSize, int outputSize, double initial_learning_rate) {
        this.outputSize = outputSize;
        this.inputSize = inputSize;

        this.weights = XavierInitializer.xavierInit2D(inputSize, outputSize);
        this.biases = XavierInitializer.xavierInit1D(outputSize);

        weightsOptimizer = new AdamOptimizer(inputSize * outputSize, initial_learning_rate, 0.9, 0.999, 1e-8);
        biasesOptimizer = new AdamOptimizer(outputSize, initial_learning_rate, 0.9, 0.999, 1e-8);
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

}