package DCGAN;

import java.util.Random;

public class DenseLayer {
    private double[][] weights;
    private double[] biases;
    double[] input;

    public DenseLayer(int inputSize, int outputSize) {
        // Initialize weights randomly
        Random rand = new Random();
        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = (double) rand.nextGaussian(); // Initialize weights with random values
            }
        }
        // Initialize biases with zeros
        for (int i = 0; i < outputSize; i++) {
            biases[i] = 0;
        }
    }

    public double[] forward(double[] input) {
        this.input = input;
        // Perform matrix multiplication of input with weights and add biases
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

    // public double[] forward_ReLU(double[] input) {
    //     this.input = input;
    //     // Perform matrix multiplication of input with weights and add biases
    //     double[] output = new double[weights[0].length];
    //     for (int j = 0; j < weights[0].length; j++) {
    //         double sum = 0;
    //         for (int i = 0; i < weights.length; i++) {
    //             sum += input[i] * weights[i][j];
    //         }
    //         output[j] = (double) Math.max(0.01 * sum + biases[j], sum + biases[j]);    //ReLU activation
    //     }
    //     return output;
    // }

    // public double[] forward_Sigmoid(double[] input) {
    //     this.input = input;
    //     // Perform matrix multiplication of input with weights and add biases
    //     double[] output = new double[weights[0].length];
    //     for (int j = 0; j < weights[0].length; j++) {
    //         double sum = 0;
    //         for (int i = 0; i < weights.length; i++) {
    //             sum += input[i] * weights[i][j];
    //         }
    //         output[j] = (double) (1 / (1 + Math.exp(-sum + biases[j]))); // Sigmoid activation
    //     }
    //     return output;
    // }

    public double[] backward(double[] outputGradient, double learningRate) {
        // Update weights and biases
        System.out.printf("Output Gradient Length %d\n", outputGradient.length);
        System.out.printf("Input Size %d Output Size %d\n", weights.length, weights[0].length);
        double[] inputGradient = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double sum = 0;
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= learningRate * outputGradient[j] * weights[i][j];
                sum += outputGradient[j] * weights[i][j];
            }
            inputGradient[i] = sum;
        }
        for (int j = 0; j < weights[1].length; j++) {
            biases[j] -= learningRate * outputGradient[j];
        }
        return inputGradient;
    }
}