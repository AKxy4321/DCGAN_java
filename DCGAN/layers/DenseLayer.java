package DCGAN.layers;

import DCGAN.XavierInitializer;

public class DenseLayer {
    private double[][] weights;
    private double[] biases;
    double[] input;

    public int outputSize, inputSize;

    public DenseLayer(int inputSize, int outputSize) {
//        weights = new double[inputSize][outputSize];
//        biases = new double[outputSize];
        this.outputSize = outputSize;
        this.inputSize = inputSize;

        this.weights = XavierInitializer.xavierInit2D(inputSize, outputSize);
        this.biases = XavierInitializer.xavierInit1D(outputSize);
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


    public void updateParameters(double[] outputGradient, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= learningRate * outputGradient[j] * input[i];
            }
        }
        for (int j = 0; j < weights[0].length; j++) {
            biases[j] -= learningRate * outputGradient[j];
        }
    }

    public void updateParameters(double[] outputGradient, double[] input, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= learningRate * outputGradient[j] * input[i];
            }
        }
        for (int j = 0; j < weights[0].length; j++) {
            biases[j] -= learningRate * outputGradient[j];
        }
    }

}
