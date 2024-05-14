package DCGAN.layers;

import DCGAN.XavierInitializer;

public class DenseLayer {
    private double[][] weights;
    private double[] biases;
    private double[] input;
    private AdamOptimizer adam;

    public int outputSize, inputSize;

    public DenseLayer(int inputSize, int outputSize, double learningRate, double beta1, double beta2, double epsilon) {
        this.outputSize = outputSize;
        this.inputSize = inputSize;

        this.weights = XavierInitializer.xavierInit2D(inputSize, outputSize);
        this.biases = XavierInitializer.xavierInit1D(outputSize);
        this.adam = new AdamOptimizer(learningRate, beta1, beta2, epsilon);
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

    public void updateParameters(double[] outputGradient) {
        double[] update = adam.update(outputGradient);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= update[i * weights[0].length + j] * input[i];
            }
        }
        for (int j = 0; j < weights[0].length; j++) {
            biases[j] -= update[weights.length * weights[0].length + j];
        }
    }


    private double[] flattenOutputGradient(double[] outputGradient) {
        double[] flatOutputGradient = new double[outputGradient.length];
        for (int i = 0; i < outputGradient.length; i++) {
            flatOutputGradient[i] = outputGradient[i];
        }
        return flatOutputGradient;
    }

    private double[] flattenInput(double[] input) {
        double[] flatInput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            flatInput[i] = input[i];
        }
        return flatInput;
    }
}
