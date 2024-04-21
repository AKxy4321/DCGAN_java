package DCGAN;

import java.util.Random;

class ConvolutionalLayer {
    double[][][] filters;
    private double[] biases;
    private double[][][] filtersGradient;
    private double[] biasesGradient;
    double[][] input;
    final public int numFilters;
    final public int filterSize;

    public ConvolutionalLayer(int filterSize, int numFilters) {
        // Initialize filters randomly
        Random rand = new Random();
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filters = new double[numFilters][filterSize][filterSize];
        biases = new double[numFilters];
        filtersGradient = new double[numFilters][filterSize][filterSize];
        biasesGradient = new double[numFilters];
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterSize; c++) {
                for (int i = 0; i < filterSize; i++) {
                    this.filters[k][c][i] = (double) rand.nextGaussian();
                }
            }
            // Initialize biases with zeros
            this.biases[k] = 0;
        }
    }

    public double[][] forward(double[][] input) {
        //output_size = (input_size - filter_size) / stride + 1
        this.input = input;
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int numFilters = this.numFilters;
        int filterSize = this.filterSize;
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;

        double[][] output = new double[numFilters][outputHeight * outputWidth];

        // Convolution operation
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    double sum = 0;
                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            sum += input[h + i][w + j] * this.filters[k][i][j];
                        }
                    }
                    output[k][h * outputWidth + w] = leakyReLU(sum + this.biases[k]);
                }
            }
        }
        return output;
    }

    public double leakyReLU(double x) {
        return x >= 0 ? x : 0.01f * x;
    }

    public double[][] backward(double[][] outputGradient, double learningRate) {
        //output_size = (input_size - filter_size) / stride + 1
        double[][] input = this.input;
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;
        int outputHeight = (int) Math.sqrt(input[0].length);
        int outputWidth = (int) Math.sqrt(input[0].length);

        // Reset gradients to zero
        for (int k = 0; k < numFilters; k++) {
            this.biasesGradient[k] = 0;
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    this.filtersGradient[k][i][j] = 0;
                }
            }
        }

//        System.out.printf("Output Gradient Depth %d Length %d\n", outputGradient.length, outputGradient[0].length);
//        System.out.printf("Input Height %d Width %d\n", inputHeight, inputWidth);
//        System.out.printf("Output Height %d Width %d\n", outputHeight, outputHeight);
        // Compute gradients
        for (int k = 0; k < numFilters; k++) {
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    for (int h = 0; h < outputHeight; h++) {
                        for (int w = 0; w < outputWidth; w++) {
                            int inputH = h + i;
                            int inputW = w + j;
                            if ((inputH >= 0 && inputH < inputHeight) && (inputW >= 0 && inputW < inputWidth)) {
                                this.filtersGradient[k][i][j] += input[inputH][inputW] * outputGradient[k][h * outputWidth + w];
                            }
                        }
                    }
                }
            }
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    this.biasesGradient[k] += outputGradient[k][h * outputWidth + w];
                }
            }
        }

        // Compute input gradients for the next layer
        double[][] inputGradient = new double[inputHeight][inputWidth];
        for (int h = 0; h < inputHeight; h++) {
            for (int w = 0; w < inputWidth; w++) {
                double sum = 0;
                for (int k = 0; k < numFilters; k++) {
                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            if (h - i >= 0 && h - i < outputHeight && w - j >= 0 && w - j < outputWidth) {
                                sum += this.filters[k][i][j] * outputGradient[k][((h - i) * outputWidth) + (w - j)];
                            }
                        }
                    }
                }
                inputGradient[h][w] = sum;
            }
        }
        // Update filters and biases
        for (int k = 0; k < numFilters; k++) {
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    this.filters[k][i][j] -= learningRate * this.filtersGradient[k][i][j];
                }
            }
            this.biases[k] -= learningRate * this.biasesGradient[k];
        }

        return inputGradient;
    }
}