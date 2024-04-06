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

    public static void main(String[] args) {
        // Example usage
        int inputSize = 2;
        int outputSize = 3;
        DenseLayer layer = new DenseLayer(inputSize, outputSize);

        double[] input = {1.0, 2.0}; // Example input
        double[] output = layer.forward(input); // Forward pass through the layer

        // Print output
        System.out.println("Output:");
        for (double val : output) {
            System.out.println(val);
        }
    }
}
