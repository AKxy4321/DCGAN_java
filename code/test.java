// Test of Forward and Backward prop

import java.util.Arrays;
import java.util.Random;

public class DenseLayer {
    private double[][] weights;
    private double[] biases;
    private double[][] weightsGradient;
    private double[] biasesGradient;

    public DenseLayer(int inputSize, int outputSize) {
        // Initialize weights randomly
        Random rand = new Random();
        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];
        weightsGradient = new double[inputSize][outputSize];
        biasesGradient = new double[outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = rand.nextGaussian(); // Initialize weights with random values
            }
        }
        // Initialize biases with zeros
        for (int i = 0; i < outputSize; i++) {
            biases[i] = 0;
        }
    }

    public double[] forward(double[] input) {
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

    public double[] backward(double[] input, double[] outputGradient, double learningRate) {
        // Compute gradients
        for (int j = 0; j < weights[0].length; j++) {
            for (int i = 0; i < weights.length; i++) {
                weightsGradient[i][j] += input[i] * outputGradient[j];
            }
            biasesGradient[j] += outputGradient[j];
        }

        // Compute input gradients for the next layer
        double[] inputGradient = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double sum = 0;
            for (int j = 0; j < weights[0].length; j++) {
                sum += weights[i][j] * outputGradient[j];
            }
            inputGradient[i] = sum;
        }

        // Update weights and biases
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= learningRate * weightsGradient[i][j];
                weightsGradient[i][j] = 0; // Reset gradients for the next iteration
            }
        }
        for (int j = 0; j < biases.length; j++) {
            biases[j] -= learningRate * biasesGradient[j];
            biasesGradient[j] = 0; // Reset gradients for the next iteration
        }

        return inputGradient;
    }

    public double computeLoss(double[] output, double[] target) {
        // Compute Mean Squared Error (MSE)
        double sumSquaredError = 0;
        for (int i = 0; i < output.length; i++) {
            double error = output[i] - target[i];
            sumSquaredError += error * error;
        }
        return sumSquaredError / output.length;
    }

    public static void main(String[] args) {
        // Example usage
        int inputSize = 2;
        int hiddenSize = 3;
        int outputSize = 2;
        double learningRate = 0.01;
        int iterations = 1000;
        // basically some value close to 0, cause exact zero change is hard to achieve
        double minLossChange = 1e-6; // Minimum change in loss to continue training

        double prevLoss = Double.MAX_VALUE;

        // Create layers
        DenseLayer layer1 = new DenseLayer(inputSize, hiddenSize);
        DenseLayer layer2 = new DenseLayer(hiddenSize, outputSize);

        // Example input
        double[] input = {1.0, 2.0};

        // Forward pass
        double[] hiddenOutput = layer1.forward(input);
        double[] finalOutput = layer2.forward(hiddenOutput);

        // Compute initial loss
        double[] target = {0.5, 0.7};
        double loss = layer2.computeLoss(finalOutput, target);
        System.out.println("Initial Loss: " + loss);

        // Training loop
        int i;
        for (i = 0; i < iterations; i++) {
            // Backward pass
            double[] outputGradient = new double[finalOutput.length];
            for (int j = 0; j < finalOutput.length; j++) {
                outputGradient[j] = finalOutput[j] - target[j];
            }
            double[] hiddenGradient = layer2.backward(hiddenOutput, outputGradient, learningRate);
            layer1.backward(input, hiddenGradient, learningRate);

            // Forward pass after training
            hiddenOutput = layer1.forward(input);
            finalOutput = layer2.forward(hiddenOutput);

            // Compute loss
            double newLoss = layer2.computeLoss(finalOutput, target);

            // Check for convergence
            if (prevLoss - newLoss < minLossChange || newLoss >= prevLoss) {
                break; // Stop training if loss doesn't decrease significantly or starts increasing
            }

            prevLoss = newLoss;
        }

        // Print final output and loss
        System.out.println("Final output:");
        System.out.println(Arrays.toString(finalOutput));
        System.out.println("Final Loss: " + prevLoss);
        System.out.println("Stopped at iteration: " + i);
    }

}
