package DCGAN;

import java.awt.image.BufferedImage;

public class TransposeConvolutionalLayer {
    double[][][][] filters;
//    private double[] biases;
    private final int stride;
    double[][][] input;
    public int numFilters;
    public int filterDepth;

    public int filterSize;
    public int inputWidth, inputHeight, inputDepth;
    public int outputWidth, outputHeight, outputDepth;
    public int padding = 0;

    public TransposeConvolutionalLayer(int inputDepth, int filterSize, int numFilters, int stride) {
        this(filterSize, numFilters, stride, 28, 28, inputDepth, 0);
    }


    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filters = new double[numFilters][inputDepth][filterSize][filterSize];
//        this.biases = new double[numFilters];
        this.filterDepth = inputDepth;

        this.filters = XavierInitializer.xavierInit4D(numFilters, filterDepth, filterSize);

        // TODO: Change to xavier initialization later, cause we aren't currently using bias
//        this.biases = UTIL.multiplyScalar(XavierInitializer.xavierInit1D(numFilters), 0);
        this.stride = stride;

        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding

        outputHeight = this.stride * (inputHeight - 1) + filterSize - 2 * padding;
        outputWidth = this.stride * (inputWidth - 1) + filterSize - 2 * padding;
        outputDepth = numFilters;
    }

    public double[][][] forward(double[][][] input) {
        this.input = input;


        double[][][] output = new double[numFilters][outputHeight][outputWidth];

        for (int oy = 0; oy < outputHeight; oy++) {
            for (int ox = 0; ox < outputWidth; ox++) {
                for (int k = 0; k < numFilters; k++) {

                    double sum = 0;
                    for (int c = 0; c < inputDepth; c++) {

                        for (int fy = 0; fy < filterSize; fy++) {
                            for (int fx = 0; fx < filterSize; fx++) {
                                int inH = oy - fy * this.stride;
                                int inW = ox - fx * this.stride;
                                if ((0 <= inH && inH < inputHeight)
                                        && (0 <= inW && inW < inputWidth)) {
                                    sum += input[c][inH][inW] * this.filters[k][c][fy][fx];
                                }
                            }
                        }
                    }

                    output[k][oy][ox] = sum;// + this.biases[k];
                }
            }
        }
        return output;
    }

    public double[][][] backward(double[][][] outputGradient) {

        // to do backward propagation for transpose conv layer,
        // you need to basically do forward propagation for outputGradient of the layer
        // using the same filters

        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];

        for (int d = 0; d < inputDepth; d++) {

            int y = -this.padding;
            for (int inputY = 0; inputY < inputHeight; y += stride, inputY++) {  // xy_stride
                int x = -this.padding;
                for (int inputX = 0; inputX < inputWidth; x += stride, inputX++) {  // xy_stride

                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int k = 0; k < numFilters; k++) {

                        for (int fy = 0; fy < filterSize; fy++) {
                            int outputY = y + fy*stride; // coordinates in the original input array coordinates
                            for (int fx = 0; fx < filterSize; fx++) {

                                int outputX = x + fx*stride;
                                if (outputY >= 0 && outputY < outputHeight && outputX >= 0 && outputX < outputWidth) {
                                    for (int fd = 0; fd < filters[0].length; fd++) {// filter depth
                                        double[][][] f = this.filters[k];

                                        //calculate the 180 degree rotated filter indices
                                        int new_fx = filterSize - 1 - fx;
                                        int new_fy = filterSize - 1 - fy;

                                        a += f[fd][new_fy][new_fx] * outputGradient[k][outputY][outputX];
                                    }
                                }
                            }
                        }
                    }
                    inputGradient[d][inputY][inputX] = a;
                }
            }
        }

        return inputGradient;
    }

    public void updateParameters(double[][][] outputGradient, double learningRate) {
        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];
//        double[] biasesGradient = new double[numFilters];


        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {

                        for (int h = 0; h < outputHeight; h++) {
                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                if ((0 <= inH && inH < inputHeight)
                                        && (0 <= inW && inW < inputWidth)) {
                                    filtersGradient[k][c][i][j] += outputGradient[k][h][w] * input[c][inH][inW];
                                }
                            }
                        }

                    }
                }
            }
//            for (int h = 0; h < outputHeight; h++) {
//                for (int w = 0; w < outputWidth; w++) {
//                    biasesGradient[k] += outputGradient[k][h][w];
//                }
//            }
        }

        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        this.filters[k][c][i][j] -= learningRate * filtersGradient[k][c][i][j];
                    }
                }
            }
//            this.biases[k] -= learningRate * biasesGradient[k];
        }
    }

    public static double calculateMSE(double[][] predictedOutput, double[][] targetOutput) {
        double mse = 0.0;

        for (int b = 0; b < predictedOutput.length; b++) {
            for (int i = 0; i < predictedOutput[0].length; i++) {
                mse += Math.pow(predictedOutput[b][i] - targetOutput[b][i], 2); // Calculate MSE
            }
        }

        mse /= (predictedOutput.length * predictedOutput.length);
        return mse;
    }
    public static double[][] calculateOutputGradient(double[][] predictedOutput, double[][] targetOutput) {
        int batchSize = predictedOutput.length;
        int outputSize = predictedOutput[0].length;
        double[][] outputGradient = new double[batchSize][outputSize];

        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < outputSize; i++) {
                outputGradient[b][i] = 2 * (predictedOutput[b][i] - targetOutput[b][i]); // Gradient of MSE loss
            }
        }

        return outputGradient;
    }

    public static void main(String[] args) {
//        output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding
        TransposeConvolutionalLayer layer = new TransposeConvolutionalLayer(2, 1, 2, 2, 2, 1, 0);
        double[][][] input = {{
                {0, 1},
                {2, 3}
        }};

        double[][][] filter = {{
                {0, 1},
                {2, 3}
        }};
        layer.filters[0] = filter;

//        double[] biases = new double[1];
//        layer.biases = biases;

        double[][][] output = layer.forward(input);
        System.out.println("Output:");
        for (int i = 0; i < output[0].length; i++) {
            for (int j = 0; j < output[0][0].length; j++) {
                System.out.print(output[0][i][j] + " ");
            }
            System.out.println();
        }

        double[][] targetOutput = {
                {0, 0, 0, 1},
                {0, 3, 2, 3},
                {0, 2, 0, 3},
                {4, 6, 6, 9}
        };


        for (int epoch = 0; epoch < 5000; epoch++) {
            double[][] res = layer.forward(input)[0];
            double[][] outputGradient = layer.calculateOutputGradient(res, targetOutput);
            double mse = layer.calculateMSE(res, targetOutput);
            System.out.println("Epoch " + (epoch + 1) + ", MSE: " + mse);
            layer.updateParameters(new double[][][]{outputGradient},0.05);


            System.out.println("Output:");
            for (int i = 0; i < output[0].length; i++) {
                for (int j = 0; j < output[0][0].length; j++) {
                    System.out.print(res[i][j] + " ");
                }
                System.out.println();
            }
        }

        UTIL.saveImage(UTIL.getBufferedImage(output), "output.png");

    }
}
/*
* To test layer:
* TransposeConvolutionalLayer layer = new TransposeConvolutionalLayer(1, 2, 1, 2);
        double[][][] input = {{
                {0, 1},
                {2, 3}
        }};

        double[][][] filter = {{
                {0, 1},
                {2, 3}
        }};
        layer.filters[0] = filter;

        double[] biases = new double[1];
        layer.biases = biases;

        double[][][] output = layer.forward(input);
        System.out.println("Output:");
        for (int i = 0; i < output[0].length; i++) {
            for (int j = 0; j < output[0][0].length; j++) {
                System.out.print(output[0][i][j] + " ");
            }
            System.out.println();
        }

*
0.0 0.0 0.0 1.0
0.0 0.0 2.0 3.0
0.0 2.0 0.0 3.0
4.0 6.0 6.0 9.0
*
        For stride one for [1,2],[3,4] and filter [1,2],[3,4] the output is:
        1.0 4.0 4.0
        6.0 20.0 16.0
        9.0 24.0 16.0

* */