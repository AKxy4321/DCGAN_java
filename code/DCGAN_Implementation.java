import java.util.Arrays;
import java.util.Random;


public class DCGAN_Implementation {

}

class Generator_Implementation{}

class Discriminator_Implementation{}

class DenseLayer {
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
        layers.DenseLayer layer1 = new layers.DenseLayer(inputSize, hiddenSize);
        layers.DenseLayer layer2 = new layers.DenseLayer(hiddenSize, outputSize);

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


class ConvolutionalLayer {
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
        layers.ConvolutionalLayer convLayer = new layers.ConvolutionalLayer(inputChannels, filterSize, numFilters);

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


class TransposeConvolutionalLayer {
    private double[][][] filters;
    private double[] biases;
    private double[][][] filtersGradient;
    private double[] biasesGradient;
    private int stride; // Stride parameter for transposed convolution

    public TransposeConvolutionalLayer(int inputChannels, int filterSize, int numFilters, int stride) {
        // Initialize filters randomly (consider using Xavier initialization for transposed convolution)
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
            biases[k] = 0;
        }
        this.stride = stride;
    }

    public double[][][] forward(double[][][] input) {
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = filters.length;
        int filterSize = filters[0][0].length;

        // Calculate output dimensions based on transposed convolution formula
        int outputHeight = stride * (inputHeight - 1) + filterSize;
        int outputWidth = stride * (inputWidth - 1) + filterSize;

        double[][][] output = new double[numFilters][outputHeight][outputWidth];

        // Transposed convolution operation
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    double sum = 0;
                    // Iterate through input with stride to perform upsampling
                    for (int c = 0; c < inputChannels; c++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                int inH = h - i * stride;
                                int inW = w - j * stride;
                                // Handle edge cases to avoid accessing outside input boundaries
                                if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth) {
                                    sum += input[c][inH][inW] * filters[k][c][i];
                                }
                            }
                        }
                    }
                    output[k][h][w] = sum + biases[k];
                }
            }
        }
        return output;
    }

    public double[][][] backward(double[][][] input, double[][][] outputGradient, double learningRate) {
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = filters.length;
        int filterSize = filters[0][0].length;

        // Calculate output dimensions from forward pass for reference
        int outputHeight = stride * (inputHeight - 1) + filterSize;
        int outputWidth = stride * (inputWidth - 1) + filterSize;

        // Reset gradients to zero
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                Arrays.fill(filtersGradient[k][c], 0);
            }
            biasesGradient[k] = 0;
        }

        // Compute gradients for filters and biases
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        for (int h = 0; h < outputHeight; h++) {

                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * stride;
                                int inW = w - j * stride;
                                // Handle edge cases
                                if (0 <= inH && inH < inputHeight - filterSize + 1 && 0 <= inW && inW < inputWidth - filterSize + 1) {

                                    filtersGradient[k][c][i] += outputGradient[k][h][w] * input[c][inH][inW];
                                }
                            }
                        }
                    }
                }
                for (int h = 0; h < outputHeight; h++) {
                    for (int w = 0; w < outputWidth; w++) {
                        biasesGradient[k] += outputGradient[k][h][w];
                    }
                }
            }
        }

        // Compute input gradients for the previous layer using transposed convolution principles
        double[][][] inputGradient = new double[inputChannels][inputHeight][inputWidth];
        for (int c = 0; c < inputChannels; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    double sum = 0;
                    for (int k = 0; k < numFilters; k++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                int outH = h + i * stride;
                                int outW = w + j * stride;
                                // Handle edge cases
                                if (outH >= 0 && outH < outputHeight && outW >= 0 && outW < outputWidth) {
                                    sum += filters[k][c][i] * outputGradient[k][outH][outW];
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

    // Other methods can be included here, similar to ConvolutionalLayer class
    public double computeLoss(double[][][] output, double[][][] target) {
        // Compute Mean Squared Error (MSE)
        double sumSquaredError = 0;
        for (int k = 0; k < output.length; k++) {
            for (int h = 0; h < output[k].length; h++) {
                for (int w = 0; w < output[k][h].length; w++) {
                    double error = output[k][h][w] - target[k][h][w];
                    sumSquaredError += error * error;
                }
            }
        }
        return sumSquaredError / (output.length * output[0].length * output[0][0].length);
    }

    public static void main(String[] args) {
        // Example usage for transposed convolution
        int inputChannels = 2; // Number of input channels
        int filterSize = 3; // Size of each filter
        int numFilters = 16; // Number of filters in transposed layer
        int inputHeight = 16; // Height of input
        int inputWidth = 4; // Width of input
        int stride = 2; // Stride for transposed convolution (controls output size)
        double learningRate = 0.01;
        int iterations = 1000;
        double minLossChange = 1e-6; // Minimum change in loss to continue training

        double prevLoss = Double.MAX_VALUE;

        // Create transposed convolutional layer
        layers.TransposeConvolutionalLayer transConvLayer = new layers.TransposeConvolutionalLayer(inputChannels, filterSize, numFilters, stride);

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
        double[][][] output = transConvLayer.forward(input);

        // Compute initial loss (random target for demonstration)
        int targetHeight = stride * (inputHeight - 1) + filterSize; // Calculate target dimensions based on stride
        int targetWidth = stride * (inputWidth - 1) + filterSize;
        double[][][] target = new double[numFilters][targetHeight][targetWidth];
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < targetHeight; h++) {
                for (int w = 0; w < targetWidth; w++) {
                    target[k][h][w] = rand.nextDouble(); // Random target values
                }
            }
        }
        double loss = transConvLayer.computeLoss(output, target);
        System.out.println("Initial Loss: " + loss);

        // Training loop
        int i;
        for (i = 0; i < iterations; i++) {
            // Compute output gradient based on mean squared error loss
            double[][][] outputGradient = new double[numFilters][targetHeight][targetWidth];
            for (int k = 0; k < numFilters; k++) {
                for (int h = 0; h < targetHeight; h++) {
                    for (int w = 0; w < targetWidth; w++) {
                        outputGradient[k][h][w] = 2 * (output[k][h][w] - target[k][h][w]) / (target.length * target[0].length * target[0][0].length); // Gradient of MSE loss
                    }
                }
            }

            // Backward pass
            double[][][] inputGradient = transConvLayer.backward(input, outputGradient, learningRate);

            // Forward pass after training (optional)
            output = transConvLayer.forward(input); // Uncomment if you want to see output after each iteration

            // Compute loss
            double newLoss = transConvLayer.computeLoss(output, target);
            System.out.println(newLoss);

            // Check for convergence
            if (Math.abs(prevLoss - newLoss) < minLossChange) {
                break; // Stop training if loss doesn't decrease significantly or starts increasing
            }

            prevLoss = newLoss;
        }

        // Print final output and loss
        System.out.println("Final output:");
        for (int k = 0; k < numFilters; k++) {
            System.out.println(Arrays.deepToString(output[k])); // Use deepToString for 3D arrays
        }
        System.out.println("Final Loss: " + prevLoss);
        System.out.println("Stopped at iteration: " + i);
    }

}