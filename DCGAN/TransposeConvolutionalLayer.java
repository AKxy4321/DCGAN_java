package DCGAN;

import java.util.Arrays;

public class TransposeConvolutionalLayer {
    double[][][][] filters;
    double[] biases;
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
        this.biases = new double[numFilters];
        this.filterDepth = inputDepth;

        this.filters = XavierInitializer.xavierInit4D(numFilters, filterDepth, filterSize);

        // TODO: Change to xavier initialization later, cause we aren't currently using bias
        this.biases = XavierInitializer.xavierInit1D(numFilters);
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
        int paddedInputHeight = inputHeight + padding * 2;
        int paddedInputWidth = inputWidth + padding * 2;

        // Pad the input with zeros
        double[][][] paddedInput = new double[inputDepth][paddedInputHeight][paddedInputWidth];
        for (int c = 0; c < inputDepth; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    paddedInput[c][h + padding][w + padding] = input[c][h][w];
                }
            }
        }

        double[][][] output = new double[numFilters][outputHeight][outputWidth];

        for (int oy = 0; oy < outputHeight; oy++) {
            for (int ox = 0; ox < outputWidth; ox++) {
                for (int k = 0; k < numFilters; k++) {

                    double sum = 0.0;
                    for (int c = 0; c < inputDepth; c++) {

                        for (int fy = 0; fy < filterSize; fy++) {
                            for (int fx = 0; fx < filterSize; fx++) {
                                // Consider padding by using valid indices within padded input
                                int inH = oy - fy * stride;
                                int inW = ox - fx * stride;
                                int effectiveInH = inH + padding;
                                int effectiveInW = inW + padding;

                                if ((0 <= effectiveInH && effectiveInH < paddedInputHeight)
                                        && (0 <= effectiveInW && effectiveInW < paddedInputWidth)) {
                                    sum += paddedInput[c][effectiveInH][effectiveInW] * this.filters[k][c][fy][fx];
                                }
                            }
                        }
                    }

                    output[k][oy][ox] = sum + this.biases[k]; // Add bias to the weighted sum
                }
            }
        }
        return output;
    }


    public double[][][] backward(double[][][] outputGradient) {

        int paddedOutputHeight = outputHeight + padding * 2;
        int paddedOutputWidth = outputWidth + padding * 2;

        // Pad the output gradient with zeros (assuming zero padding during forward pass)
        double[][][] paddedOutput = new double[numFilters][paddedOutputHeight][paddedOutputWidth];
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    paddedOutput[k][h + padding][w + padding] = outputGradient[k][h][w];
                }
            }
        }

        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];

        for (int d = 0; d < inputDepth; d++) {

            for (int inputY = 0; inputY < inputHeight; inputY++) {
                for (int inputX = 0; inputX < inputWidth; inputX++) {

                    double a = 0.0;
                    for (int k = 0; k < numFilters; k++) {

                        for (int fy = 0; fy < filterSize; fy++) {
                            for (int fx = 0; fx < filterSize; fx++) {
                                // Consider padding by using valid indices within padded output
                                int outputY = inputY + fy * stride;
                                int outputX = inputX + fx * stride;
                                int effectiveOutputY = outputY + padding;
                                int effectiveOutputX = outputX + padding;

                                if ((0 <= effectiveOutputY && effectiveOutputY < paddedOutputHeight)
                                        && (0 <= effectiveOutputX && effectiveOutputX < paddedOutputWidth)) {
                                    a += this.filters[k][d][filterSize - 1 - fy][filterSize - 1 - fx] * paddedOutput[k][effectiveOutputY][effectiveOutputX];
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


    public void updateParameters(double[][][] outputGradient, double[][][] input, double learningRate) {
        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];
        double[] biasGradient = new double[numFilters]; // to store bias gradients

        // Calculate filter gradients (same as before)
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {

                        for (int h = 0; h < outputHeight; h++) {
                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                int effectiveInH = inH + padding; // Consider padding
                                int effectiveInW = inW + padding;

                                if ((0 <= effectiveInH && effectiveInH < inputHeight + 2 * padding)
                                        && (0 <= effectiveInW && effectiveInW < inputWidth + 2 * padding)) {
                                    filtersGradient[k][c][i][j] += outputGradient[k][h][w] * input[c][effectiveInH - padding][effectiveInW - padding];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Calculate bias gradients
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    biasGradient[k] += outputGradient[k][h][w]; // Sum over output
                }
            }
            biasGradient[k] /= (outputWidth * outputHeight); // Average gradient
        }

        // Update filters and biases
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        this.filters[k][c][i][j] -= learningRate * filtersGradient[k][c][i][j];
                    }
                }
            }
            this.biases[k] -= learningRate * biasGradient[k];
        }
    }


    public void updateParameters(double[][][] outputGradient, double learningRate) {
        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];
        double[] biasGradient = new double[numFilters]; // to store bias gradients

        // Calculate filter gradients (same as before)
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {

                        for (int h = 0; h < outputHeight; h++) {
                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;
                                int effectiveInH = inH + padding; // Consider padding
                                int effectiveInW = inW + padding;

                                if ((0 <= effectiveInH && effectiveInH < inputHeight + 2 * padding)
                                        && (0 <= effectiveInW && effectiveInW < inputWidth + 2 * padding)) {
                                    filtersGradient[k][c][i][j] += outputGradient[k][h][w] * input[c][effectiveInH - padding][effectiveInW - padding];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Calculate bias gradients
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    biasGradient[k] += outputGradient[k][h][w]; // Sum over output
                }
            }
            biasGradient[k] /= (outputWidth * outputHeight); // Average gradient
        }

        // Update filters and biases
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        this.filters[k][c][i][j] -= learningRate * filtersGradient[k][c][i][j];
                    }
                }
            }
            this.biases[k] -= learningRate * biasGradient[k];
        }
    }


    public static void main(String[] args) {

        DenseLayer dense = new DenseLayer(100, 100);
        SigmoidLayer sigmoidLayer = new SigmoidLayer();
        DenseLayer dense2 = new DenseLayer(dense.outputSize, 2*2*1);
        LeakyReLULayer leakyReLULayer = new LeakyReLULayer(0.01);
//        output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding
        TransposeConvolutionalLayer tconv = new TransposeConvolutionalLayer(2, 5, 2, 2, 2, 1, 0);
        TransposeConvolutionalLayer tconv2 = new TransposeConvolutionalLayer(1, 1, 1, tconv.outputWidth, tconv.outputHeight, tconv.outputDepth, 0);

//        double[][] targetOutput = {
//                {0, 0, 0, 0},
//                {0, 3, 2, 3},
//                {0, 2, 0, 3},
//                {4, 6, 6, 9}
//        };

//        double[][] targetOutput = {
//                {0, 0, 0, 0},
//                {0, 0, 0, 0},
//                {0, 0, 0, 0},
//                {0, 0, 0, 0},
//        };

        double[][] targetOutput = {
                {10, 10, 10, 10},
                {10, 10, 10, 10},
                {10, 10, 10, 10},
                {10, 10, 10, 10},
        };
//        targetOutput = UTIL.multiplyScalar(targetOutput, 10);


        double learning_rate = 0.0001;

        for (int epoch = 0; epoch < 5000; epoch++) {
            double[] input = XavierInitializer.xavierInit1D(dense.inputSize); // {1, 1, 1, 1, 1};

            //forward pass
            double[] dense_output = dense.forward(input);
            double[] activation_function_output = sigmoidLayer.forward(dense_output);
            double[] dense2_output = dense2.forward(activation_function_output);
            double[] leaky_output = leakyReLULayer.forward(dense2_output);
            double[][][] tconv_output = tconv.forward(UTIL.unflatten(leaky_output, tconv.inputDepth, tconv.inputHeight, tconv.inputWidth));
            double[][][] tconv2_output = tconv2.forward(tconv_output);

            double[][] outputGradient = new double[tconv2_output[0].length][tconv2_output[0][0].length];
            UTIL.calculateGradientRMSE(outputGradient, tconv2_output[0], targetOutput);
            double mse = UTIL.lossRMSE(tconv2_output[0], targetOutput);


            System.out.println("Epoch " + (epoch + 1) + ", RMSE: " + mse);

            /**
             * the backward function for all layers calculates the input gradient for that layer and returns it.
             * updateParameters function for each layer updates its parameters by calculating the gradient of
             * its weight using the ouptut gradients for that layer. I have to pass in the output gradient of that
             * layer to calculate the updated weights. The input gradient for the next layer is the output gradient for the current layer.
             *
             * tconv2_in_gradient is the input gradient for tconv2, not the output gradient.
             * Output gradient for tconv2 is the  MSE error gradient
             * */

            // backward pass
            double[][][] tconv2_in_gradient_tconv_out_gradient = tconv2.backward(new double[][][]{outputGradient});
            double[][][] tconv_in_gradient_dense2_out_gradient = tconv.backward(tconv2_in_gradient_tconv_out_gradient);
            double[] leakyrelu_in_gradient_dense2_out_gradient = leakyReLULayer.backward(UTIL.flatten(tconv_in_gradient_dense2_out_gradient));
            double[] dense2_in_gradient_lu_out_gradient = dense2.backward(leakyrelu_in_gradient_dense2_out_gradient);
            double[] lu_in_gradient_dense_out_gradient = sigmoidLayer.backward(dense2_in_gradient_lu_out_gradient);
            double[] dense_in_gradient = dense.backward(lu_in_gradient_dense_out_gradient);

            dense.updateParameters(lu_in_gradient_dense_out_gradient, learning_rate);
            dense2.updateParameters(leakyrelu_in_gradient_dense2_out_gradient, learning_rate);
            tconv.updateParameters(tconv2_in_gradient_tconv_out_gradient, learning_rate);
            tconv2.updateParameters(new double[][][]{outputGradient}, learning_rate);

            System.out.println("Sum of values in each gradient :");
            System.out.println("Dense input gradient: " + Arrays.stream(dense_in_gradient).sum());
            System.out.println("Leaky input gradient: " + Arrays.stream(lu_in_gradient_dense_out_gradient).sum());
            System.out.println("Dense2 input gradient: " + Arrays.stream(dense2_in_gradient_lu_out_gradient).sum());
            System.out.println("Tconv input gradient: " + Arrays.stream(UTIL.flatten(tconv_in_gradient_dense2_out_gradient)).sum());
            System.out.println("Tconv2 input gradient: " + Arrays.stream(UTIL.flatten(tconv2_in_gradient_tconv_out_gradient[0])).sum());


            if (epoch % 5 == 0) {
                double[][] output = tconv2_output[0];
                System.out.println("Output:");
                for (int i = 0; i < output.length; i++) {
                    for (int j = 0; j < output[0].length; j++) {
                        System.out.print(output[i][j] + ":" + targetOutput[i][j] + " ");
                    }
                    System.out.println();
                }
            }
        }
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