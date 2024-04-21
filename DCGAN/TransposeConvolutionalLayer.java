package DCGAN;

public class TransposeConvolutionalLayer {
    double[][][][] filters;
    private double[] biases;
    private double[][][][] filtersGradient;
    private double[] biasesGradient;
    private final int stride;
    double[][][] input;
    public int numFilters;
    public int filterSize;

    public TransposeConvolutionalLayer(int numFiltersPrev, int filterSize, int numFilters, int stride) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filters = new double[numFilters][numFiltersPrev][filterSize][filterSize];
        this.biases = new double[numFilters];
        this.filtersGradient = new double[numFilters][numFiltersPrev][filterSize][filterSize];
        this.biasesGradient = new double[numFilters];
        this.filters = XavierInitializer.xavierInit4D(numFilters, numFiltersPrev, filterSize);
        this.biases = XavierInitializer.xavierInit1D(numFilters);
        this.stride = stride;
    }

    public double[][][] forward(double[][][] input) {
        this.input = input;
        int inputChannels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = this.filters.length;
        int filterSize = this.filters[0][0].length;

        int outputHeight = this.stride * (inputHeight - 1) + filterSize;
        int outputWidth = this.stride * (inputWidth - 1) + filterSize;

        double[][][] output = new double[numFilters][outputHeight][outputWidth];

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
}
