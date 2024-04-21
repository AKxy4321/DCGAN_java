package DCGAN;

import java.util.Arrays;
import java.util.Random;

public class TransposeConvolutionalLayer {
    double[][][][] filters; // filterIndex, channelIndex, heightIndex, widthIndex
    private double[] biases;
    private double[][][][] filtersGradient;
    private double[] biasesGradient;
    private int stride;
    double[][][] input;
    public int numFilters;
    public int filterSize;

    public TransposeConvolutionalLayer(int numFiltersPrev, int filterSize, int numFilters, int stride) {
        Random rand = new Random();
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filters = new double[numFilters][numFiltersPrev][filterSize][filterSize];
        this.biases = new double[numFilters];
        this.filtersGradient = new double[numFilters][numFiltersPrev][filterSize][filterSize];
        this.biasesGradient = new double[numFilters];
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < numFiltersPrev; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++){
                        this.filters[k][c][i][j] = (double) rand.nextGaussian();
                    }
                }
            }
            this.biases[k] = 0;
        }
        this.stride = stride;
    }

    public double[][][] forward(double[][][] input) {
        this.input = input;
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;

        // Calculate output dimensions based on transposed convolution formula
        int outputHeight = this.stride * (inputHeight - 1) + filterSize;
        int outputWidth = this.stride * (inputWidth - 1) + filterSize;

        double[][][] output = new double[numFilters][outputHeight][outputWidth];

        // Transposed convolution operation
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    double sum = 0;
                    for (int c = 0; c < inputChannels; c++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                if ((0 <= inH && inH < inputHeight)
                                        && (0 <= inW && inW < inputWidth)) {
                                    sum += input[c][inH][inW] * this.filters[k][c][i][j];
                                }
                            }
                        }
                    }
                    output[k][h][w] = sum + this.biases[k];
                }
            }
        }
        return output;
    }

    public double[][][] backward(double[][][] outputGradient, double learningRate) {
        double[][][] input = this.input;
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;

        int outputHeight = this.stride * (inputHeight - 1) + filterSize;
        int outputWidth = this.stride * (inputWidth - 1) + filterSize;
//        System.out.printf("Input Depth %d Height %d Width %d\n", inputChannels, inputHeight, inputWidth);
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i =0;i<filterSize;i++){
                    for (int j = 0; j < filterSize; j++) {
                        this.filtersGradient[k][c][i][j] = 0;
                    }
            }
            }
            this.biasesGradient[k] = 0;
        }

        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        for (int h = 0; h < outputHeight; h++) {

                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                // Handle edge cases
                                if ((0 <= inH && inH < inputHeight - filterSize - 1)
                                        && (0 <= inW && inW < inputWidth - filterSize - 1)) {
                                    this.filtersGradient[k][c][i][j] += outputGradient[k][h][w] * input[c][inH][inW];
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

        double[][][] inputGradient = new double[inputChannels][inputHeight][inputWidth];
        for (int c = 0; c < inputChannels; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    double sum = 0;
                    for (int k = 0; k < numFilters; k++) {
                        for (int i = 0; i < filterSize; i++) {
                            for (int j = 0; j < filterSize; j++) {
                                int outH = h + i * this.stride;
                                int outW = w + j * this.stride;
                                if (outH >= 0 && outH < outputHeight && outW >= 0 && outW < outputWidth) {
                                    sum += this.filters[k][c][i][j] * outputGradient[k][outH][outW];
                                }
                            }
                        }
                    }
                    inputGradient[c][h][w] = sum;
                }
            }
        }

        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        this.filters[k][c][i][j] -= learningRate * this.filtersGradient[k][c][i][j];
                    }
                }
            }
            this.biases[k] -= learningRate * this.biasesGradient[k];
        }

        return inputGradient;
    }

    public double[][][] unflattenArray(double[] input, int numFilters, int outputHeight, int outputWidth) {
        double[][][] output = new double[numFilters][outputHeight][outputWidth];
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

    public static void main(String[] args) {
        int numFiltersPrev = 3;
        int filterSize = 2;
        int numFilters = 1;
        int stride = 2;
        TransposeConvolutionalLayer layer = new TransposeConvolutionalLayer(numFiltersPrev, filterSize, numFilters,
                stride);

        double[][][] input = { // new double[1][2][2]
                { { 1, 2 },
                  { 3, 4 }
                } };
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int outputHeight = stride * (inputHeight - 1) + filterSize;
        int outputWidth = stride * (inputWidth - 1) + filterSize;
        double[][][] targetOutput = //new double[1][outputHeight][outputWidth]
        {{{1f, 1f, 1f}, {1f, 1f, 1f}, {1f, 1f, 1f}}};
        // layer.filters[0][0][0][0] = 1;
        // layer.filters[0][0][0][1] = 2;
        // layer.filters[0][0][1][0] = 3;
        // layer.filters[0][0][1][1] = 4;


        double[][][] output = layer.forward(input);
        System.out.println(Arrays.deepToString(layer.filters));

        System.out.println(Arrays.deepToString(output));

        layer.backward(output, outputWidth);
    }
}