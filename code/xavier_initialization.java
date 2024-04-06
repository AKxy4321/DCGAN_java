// Initialising Random Values of Filters using Xavier Initialization

import java.util.Random;

public class XavierInitialization {
    public static double[][][] xavierInitialization(int inputChannels, int outputChannels, int filterSize) {
        double variance = 2.0 / (inputChannels + outputChannels); // Compute variance for Xavier initialization
        Random rand = new Random();
        double[][][] weights = new double[filterSize][filterSize][inputChannels];
        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                for (int k = 0; k < inputChannels; k++) {
                    weights[i][j][k] = rand.nextGaussian() * Math.sqrt(variance); // Generate random values from Gaussian distribution with zero mean and computed variance
                }
            }
        }
        return weights;
    }

    public static void main(String[] args) {
        int inputChannels = 3; // Number of input channels
        int outputChannels = 64; // Number of output channels
        int filterSize = 3; // Size of the filter (assuming square filter)

        // Generate Xavier initialized weights
        double[][][] weights = xavierInitialization(inputChannels, outputChannels, filterSize);

        // Print the generated weights (for demonstration)
        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                for (int k = 0; k < inputChannels; k++) {
                    System.out.printf("%.4f ", weights[i][j][k]);
                }
                System.out.println();
            }
            System.out.println();
        }
    }
}
