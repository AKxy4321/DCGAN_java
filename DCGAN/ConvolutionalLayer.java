package DCGAN;

import java.util.Random;

class ConvolutionalLayer {
    double[][][] filters;
    private double[] biases;
    double[][] input;
    final public int numFilters;
    final public int filterSize;

    public ConvolutionalLayer(int filterSize, int numFilters) {
        Random rand = new Random();
        this.numFilters = numFilters;
        this.filterSize = filterSize;
//        this.filters = new double[numFilters][filterSize][filterSize];
//        this.biases = new double[numFilters];
        this.filters = XavierInitializer.xavierInit3D(numFilters, filterSize, filterSize);
        this.biases = XavierInitializer.xavierInit1D(numFilters);
    }

    public double[][] forward(double[][] input) {
        this.input = input;
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int numFilters = this.numFilters;
        int filterSize = this.filterSize;
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;

        double[][] output = new double[numFilters][outputHeight * outputWidth];

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

    public double[][] backward(double[][] outputGradient) {
        double[][] input = this.input;
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;
        int outputHeight = (int) Math.sqrt(input[0].length);
        int outputWidth = (int) Math.sqrt(input[0].length);


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
        return inputGradient;
    }

    double updateParameters(double[][] outputGradient, double learningRate) {
        double[][] input = this.input;
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;
        int outputHeight = (int) Math.sqrt(input[0].length);
        int outputWidth = (int) Math.sqrt(input[0].length);

        double[][][] filtersGradient = new double[numFilters][filterSize][filterSize];
        double[] biasesGradient = new double[numFilters];

        for (int k = 0; k < numFilters; k++) {
            biasesGradient[k] = 0;
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    filtersGradient[k][i][j] = 0;
                }
            }
        }

        for (int k = 0; k < numFilters; k++) {
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    for (int h = 0; h < outputHeight; h++) {
                        for (int w = 0; w < outputWidth; w++) {
                            int inputH = h + i;
                            int inputW = w + j;
                            if ((inputH >= 0 && inputH < inputHeight) && (inputW >= 0 && inputW < inputWidth)) {
                                filtersGradient[k][i][j] += input[inputH][inputW] * outputGradient[k][h * outputWidth + w];
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

        for (int k = 0; k < this.numFilters; k++) {
            for (int i = 0; i < this.filterSize; i++) {
                for (int j = 0; j < this.filterSize; j++) {
                    this.filters[k][i][j] -= learningRate * filtersGradient[k][i][j];
                }
            }
            this.biases[k] -= learningRate * biasesGradient[k];
        }
        return 0;
    }
}