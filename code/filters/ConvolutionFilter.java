import java.util.Random;
import static xavier_initialization.xavierInitialization;
public class ConvolutionFilter {
    private double[][][] weights;
    private double bias;

    public ConvolutionFilter(int inputChannels, int outputChannels, int filterSize) {
        // Initialize weights using Xavier initialization
        weights = xavierInitialization(inputChannels, outputChannels, filterSize);
        // Initialize bias to zero
        bias = 0.0;
    }

    // Perform convolution operation on the input
    public double[][] convolve(double[][] input) {
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int filterSize = weights.length;
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;

        // Initialize output feature map
        double[][] output = new double[outputHeight][outputWidth];

        // Perform convolution
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                for (int k = 0; k < weights.length; k++) {
                    for (int l = 0; l < weights[0].length; l++) {
                        for (int m = 0; m < weights[0][0].length; m++) {
                            output[i][j] += input[i + k][j + l] * weights[k][l][m];
                        }
                    }
                }
                output[i][j] += bias; // Add bias
            }
        }

        return output;
    }

        public static void main(String[] args) {
        // Example usage
        int inputChannels = 3;
        int outputChannels = 1;
        int filterSize = 3;
        ConvolutionFilter filter = new ConvolutionFilter(inputChannels, outputChannels, filterSize);

        // Example input (5x5 grayscale image)
        double[][] input = {
            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25}
        };

        // Perform convolution
        double[][] output = filter.convolve(input);

        // Print output (feature map)
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                System.out.printf("%.2f ", output[i][j]);
            }
            System.out.println();
        }
    }
}
