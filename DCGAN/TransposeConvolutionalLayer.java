package DCGAN;

import java.util.Arrays;
import java.util.Random;

public class TransposeConvolutionalLayer {
    private float[][][] filters;
    private float[] biases;
    private float[][][] filtersGradient;
    private float[] biasesGradient;
    private int stride; // Stride parameter for transposed convolution
    float[][][] input;

    public TransposeConvolutionalLayer(int inputChannels, int filterSize, int numFilters, int stride) {
        // Initialize filters randomly (consider using Xavier initialization for transposed convolution)
        Random rand = new Random();
        this.filters = new float[numFilters][inputChannels][filterSize];
        this.biases = new float[numFilters];
        this.filtersGradient = new float[numFilters][inputChannels][filterSize];
        this.biasesGradient = new float[numFilters];
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    this.filters[k][c][i] = (float) rand.nextGaussian(); // Initialize filters with random values
                }
            }
            this.biases[k] = 0;
        }
        this.stride = stride;
    }

    public float[][][] forward_ReLU(float[][][] input) {
        this.input = input;
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;

        // Calculate output dimensions based on transposed convolution formula
        int outputHeight = this.stride * (inputHeight - 1) + filterSize;
        int outputWidth = this.stride * (inputWidth - 1) + filterSize;

        float[][][] output = new float[numFilters][outputHeight][outputWidth];

        // Transposed convolution operation
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    float sum = 0;
                    // Iterate through input with stride to perform upsampling
                    for (int c = 0; c < inputChannels; c++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                // Handle edge cases to avoid accessing outside input boundaries
                                if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth) {
                                    sum += input[c][inH][inW] * this.filters[k][c][i];
                                }
                            }
                        }
                    }
                    output[k][h][w] = Math.max(0.01f * sum + this.biases[k], sum + this.biases[k]);
                }
            }
        }
        return output;
    }

    public float[][][] forward_tanh(float[][][] input) {
        this.input = input;
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;

        // Calculate output dimensions based on transposed convolution formula
        int outputHeight = this.stride * (inputHeight - 1) + filterSize;
        int outputWidth = this.stride * (inputWidth - 1) + filterSize;

        float[][][] output = new float[numFilters][outputHeight][outputWidth];

        // Transposed convolution operation
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    float sum = 0;
                    // Iterate through input with stride to perform upsampling
                    for (int c = 0; c < inputChannels; c++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                // Handle edge cases to avoid accessing outside input boundaries
                                if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth) {
                                    sum += input[c][inH][inW] * this.filters[k][c][i];
                                }
                            }
                        }
                    }
                    output[k][h][w] = (float) Math.tanh(sum + this.biases[k]);
                }
            }
        }
        return output;
    }

    public float[][][] backward(float[][][] outputGradient, float learningRate) {
        float[][][] input = this.input;
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;

        // Calculate output dimensions from forward pass for reference
        int outputHeight = this.stride * (inputHeight - 1) + filterSize;
        int outputWidth = this.stride * (inputWidth - 1) + filterSize;

        // Reset gradients to zero
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                Arrays.fill(this.filtersGradient[k][c], 0);
            }
            this.biasesGradient[k] = 0;
        }

        // Compute gradients for filters and biases
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        for (int h = 0; h < outputHeight; h++) {

                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                // Handle edge cases
                                if (0 <= inH && inH < inputHeight - filterSize + 1 && 0 <= inW && inW < inputWidth - filterSize + 1) {

                                    this.filtersGradient[k][c][i] += outputGradient[k][h][w] * input[c][inH][inW];
                                }
                            }
                        }
                    }
                }
                for (int h = 0; h < outputHeight; h++) {
                    for (int w = 0; w < outputWidth; w++) {
                        this.biasesGradient[k] += outputGradient[k][h][w];
                    }
                }
            }
        }

        // Compute input gradients for the previous layer using transposed convolution principles
        float[][][] inputGradient = new float[inputChannels][inputHeight][inputWidth];
        for (int c = 0; c < inputChannels; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    float sum = 0;
                    for (int k = 0; k < numFilters; k++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                int outH = h + i * this.stride;
                                int outW = w + j * this.stride;
                                // Handle edge cases
                                if (outH >= 0 && outH < outputHeight && outW >= 0 && outW < outputWidth) {
                                    sum += this.filters[k][c][i] * outputGradient[k][outH][outW];
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
                    this.filters[k][c][i] -= learningRate * this.filtersGradient[k][c][i];
                }
            }
            this.biases[k] -= learningRate * this.biasesGradient[k];
        }

        return inputGradient;
    }

    public float[][][] unflattenArray(float[] input, int numFilters, int outputHeight, int outputWidth) {
        float[][][] output = new float[numFilters][outputHeight][outputWidth];
        int index = 0;
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    output[k][h][w] = input[index++];
                }
            }
        }
        return output;
    }
}