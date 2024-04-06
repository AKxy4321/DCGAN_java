import java.util.Arrays;
import java.util.Random;

public class ConvolutionalLayer {
    private double[][][] filters;
    private double[] biases;
    private double[][][] filtersGradient;
    private double[] biasesGradient;

    public ConvolutionalLayer(int inputChannels, int filterSize, int numFilters) {
        // Initialize filters randomly
        Random rand = new Random();
        filters = new double[numFilters][inputChannels][filterSize];
        biases = new double[numFilters];
        filtersGradient = new double[numFilters][inputChannels][filterSize];
        biasesGradient = new double[numFilters];
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    filters[k][c][i] = rand.nextGaussian(); // Initialize filters with random values
                }
            }
            // Initialize biases with zeros
            biases[k] = 0;
        }
    }

    public double[][] forward(double[][][] input) {
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = filters.length;
        int filterSize = filters[0][0].length;
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;

        double[][] output = new double[numFilters][outputHeight * outputWidth];

        // Convolution operation
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    double sum = 0;
                    for (int c = 0; c < inputChannels; c++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                sum += input[c][h + i][w + j] * filters[k][c][i];
                            }
                        }
                    }
                    output[k][h * outputWidth + w] = sum + biases[k];
                }
            }
        }
        return output;
    }


    public double[][][] backward(double[][][] input, double[][] outputGradient, double learningRate) {
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = filters.length;
        int filterSize = filters[0][0].length;
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;

        // Reset gradients to zero
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                Arrays.fill(filtersGradient[k][c], 0);
            }
            biasesGradient[k] = 0;
        }

        // Compute gradients
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        for (int h = 0; h < outputHeight; h++) {
                            for (int w = 0; w < outputWidth; w++) {
                                filtersGradient[k][c][i] += input[c][h + i][w + j] * outputGradient[k][h * outputWidth + w];
                            }
                        }
                    }
                }
            }
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    biasesGradient[k] += outputGradient[k][h * outputWidth + w];
                }
            }
        }

        // Compute input gradients for the next layer
        double[][][] inputGradient = new double[inputChannels][inputHeight][inputWidth];
        for (int c = 0; c < inputChannels; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    double sum = 0;
                    for (int k = 0; k < numFilters; k++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                if (h - i >= 0 && h - i < outputHeight && w - j >= 0 && w - j < outputWidth) {
                                    sum += filters[k][c][i] * outputGradient[k][((h - i) * outputWidth) + (w - j)];
                                }
                            }
                        }
                    }
                    inputGradient[c][h][w] = sum;
                }
            }
        }

        // Update filters and biases
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    filters[k][c][i] -= learningRate * filtersGradient[k][c][i];
                }
            }
            biases[k] -= learningRate * biasesGradient[k];
        }

        return inputGradient;
    }



    public double computeLoss(double[][] output, double[][] target) {
        // Compute Mean Squared Error (MSE)
        double sumSquaredError = 0;
        for (int k = 0; k < output.length; k++) {
            for (int i = 0; i < output[0].length; i++) {
                double error = output[k][i] - target[k][i];
                sumSquaredError += error * error;
            }
        }
        return sumSquaredError / (output.length * output[0].length);
    }

    public static void main(String[] args) {
        // Example usage
        int inputChannels = 3; // Number of input channels (e.g., RGB)
        int filterSize = 3; // Size of each filter
        int numFilters = 2; // Number of filters
        int inputHeight = 5; // Height of input
        int inputWidth = 5; // Width of input
        double learningRate = 0.01;
        int iterations = 1000;
        double minLossChange = 1e-6; // Minimum change in loss to continue training

        double prevLoss = Double.MAX_VALUE;

        // Create convolutional layer
        ConvolutionalLayer convLayer = new ConvolutionalLayer(inputChannels, filterSize, numFilters);

        // Example input (randomly generated)
        double[][][] input = new double[inputChannels][inputHeight][inputWidth];
        Random rand = new Random();
        for (int c = 0; c < inputChannels; c++) {
            for (int i = 0; i < inputHeight; i++) {
                for (int j = 0; j < inputWidth; j++) {
                    input[c][i][j] = rand.nextDouble(); // Random input values
                }
            }
        }

        // Forward pass
        double[][] output = convLayer.forward(input);

        // Compute initial loss (random target for demonstration)
        double[][] target = new double[numFilters][output[0].length];
        for (int k = 0; k < numFilters; k++) {
            for (int i = 0; i < output[0].length; i++) {
                target[k][i] = rand.nextDouble(); // Random target values
            }
        }
        double loss = convLayer.computeLoss(output, target);
        System.out.println("Initial Loss: " + loss);

        // Training loop
        int i;
        for (i = 0; i < iterations; i++) {
            // Compute output gradient based on mean squared error loss
            double[][] outputGradient = new double[numFilters][output[0].length];
            for (int k = 0; k < numFilters; k++) {
                for (int j = 0; j < output[0].length; j++) {
                    outputGradient[k][j] = 2 * (output[k][j] - target[k][j]) / (output[0].length); // Gradient of MSE loss
                }
            }

            // Backward pass
            double[][][] inputGradient = convLayer.backward(input, outputGradient, learningRate);

            // Forward pass after training
            output = convLayer.forward(input);

            // Compute loss
            double newLoss = convLayer.computeLoss(output, target);

            // Check for convergence
            if (Math.abs(prevLoss - newLoss) < minLossChange) {
                break; // Stop training if loss doesn't decrease significantly or starts increasing
            }

            prevLoss = newLoss;
        }

        // Print final output and loss
        System.out.println("Final output:");
        for (int k = 0; k < numFilters; k++) {
            System.out.println(Arrays.toString(output[k]));
        }
        System.out.println("Final Loss: " + prevLoss);
        System.out.println("Stopped at iteration: " + i);
    }
}
