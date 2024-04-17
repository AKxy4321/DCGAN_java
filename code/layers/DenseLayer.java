package layers;

import java.util.Random;

public class DenseLayer {
    private double[][] weights;
    private double[] biases;

    public DenseLayer(int inputSize, int outputSize) {
        // Initialize weights randomly
        Random rand = new Random();
        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];
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

    public double[] backward(double[] outputGradient, double learningRate) {
        // Update weights and biases
        double[] inputGradient = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double sum = 0;
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= learningRate * outputGradient[j] * weights[i][j];
                sum += outputGradient[j] * weights[i][j];
            }
            inputGradient[i] = sum;
        }
        for (int j = 0; j < weights[0].length; j++) {
            biases[j] -= learningRate * outputGradient[j];
        }
        return inputGradient;
    }

    public static void main(String[] args) {
        int inputSize = 2;
        int hiddenSize = 4; // Hidden layer size
        int outputSize = 1;

        // Create two dense layers
        DenseLayer layer1 = new DenseLayer(inputSize, hiddenSize);
        DenseLayer layer2 = new DenseLayer(hiddenSize, outputSize);

        // Generate random input data
        Random rand = new Random();
        double[] input = {rand.nextDouble(), rand.nextDouble()}; // Random input

        // Generate random target output data
        double[] targetOutput = {rand.nextDouble()}; // Random target output

        // Training parameters
        double learningRate = 0.01;
        int epochs = 1000;

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            double[] hiddenOutput = layer1.forward(input);
            double[] output = layer2.forward(hiddenOutput);

            // Calculate loss
            double loss = Math.pow(output[0] - targetOutput[0], 2); // Squared error loss

            // Backward pass for second layer
            double[] outputGradient = {2 * (output[0] - targetOutput[0])}; // Gradient of squared error loss
            double[] hiddenGradient = layer2.backward(outputGradient, learningRate);

            // Backward pass for first layer
            layer1.backward(hiddenGradient, learningRate);

            // Print loss every few epochs
            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + ", Loss: " + String.format("%.10f", loss));
            }
        }

        // Test the trained network with new input
        double[] testInput = {0.5, 0.8}; // New input
        double[] hiddenOutput = layer1.forward(testInput);
        double[] testOutput = layer2.forward(hiddenOutput);

        // Print test output
        System.out.println("Test Output:");
        System.out.println(testOutput[0]);
    }


}