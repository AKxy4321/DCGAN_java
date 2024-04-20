package DCGAN;

import java.util.Random;

public class DenseLayer {
    private float[][] weights;
    private float[] biases;
    float[] input;

    public DenseLayer(int inputSize, int outputSize) {
        // Initialize weights randomly
        Random rand = new Random();
        weights = new float[inputSize][outputSize];
        biases = new float[outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = (float) rand.nextGaussian(); // Initialize weights with random values
            }
        }
        // Initialize biases with zeros
        for (int i = 0; i < outputSize; i++) {
            biases[i] = 0;
        }
    }

    public float[] forward(float[] input) {
        this.input = input;
        // Perform matrix multiplication of input with weights and add biases
        float[] output = new float[weights[0].length];
        for (int j = 0; j < weights[0].length; j++) {
            float sum = 0;
            for (int i = 0; i < weights.length; i++) {
                sum += input[i] * weights[i][j];
            }
            output[j] = sum + biases[j];
        }
        return output;
    }

    // public float[] forward_ReLU(float[] input) {
    //     this.input = input;
    //     // Perform matrix multiplication of input with weights and add biases
    //     float[] output = new float[weights[0].length];
    //     for (int j = 0; j < weights[0].length; j++) {
    //         float sum = 0;
    //         for (int i = 0; i < weights.length; i++) {
    //             sum += input[i] * weights[i][j];
    //         }
    //         output[j] = (float) Math.max(0.01 * sum + biases[j], sum + biases[j]);    //ReLU activation
    //     }
    //     return output;
    // }

    // public float[] forward_Sigmoid(float[] input) {
    //     this.input = input;
    //     // Perform matrix multiplication of input with weights and add biases
    //     float[] output = new float[weights[0].length];
    //     for (int j = 0; j < weights[0].length; j++) {
    //         float sum = 0;
    //         for (int i = 0; i < weights.length; i++) {
    //             sum += input[i] * weights[i][j];
    //         }
    //         output[j] = (float) (1 / (1 + Math.exp(-sum + biases[j]))); // Sigmoid activation
    //     }
    //     return output;
    // }

    public float[] backward(float[] outputGradient, float learningRate) {
        // Update weights and biases
        System.out.printf("Output Gradient Length %d\n", outputGradient.length);
        System.out.printf("Input Size %d Output Size %d\n", weights.length, weights[0].length);
        float[] inputGradient = new float[weights.length];
        for (int i = 0; i < weights.length; i++) {
            float sum = 0;
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