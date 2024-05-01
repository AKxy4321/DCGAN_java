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
    public int output_padding = 0; // for now we don't care about output_padding

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
        /**
         * Consider implementing the convolution by multiplying matrices. Given an input vector x and a weight
         * matrix W, the forward propagation function of the convolution can be implemented by multiplying its
         * input with the weight matrix and outputting a vector y = W x Since backpropagation follows the
         * chain rule and delta_x y =  W , the backpropagation function of the convolution can be implemented by
         * multiplying its input with the transposed weight matrix W . Therefore, the transposed convolutional
         * layer can just exchange the forward propagation function and the backpropagation function of the
         * convolutional layer: its forward propagation and backpropagation functions multiply their input vector
         * with W and W, respectively.
         * */

//        int paddedOutputHeight = outputHeight + output_padding * 2;
//        int paddedOutputWidth = outputWidth + output_padding * 2;

//        // flipping the filters
//        double[][][][] flipped_filters = new double[numFilters][filterDepth][filterSize][filterSize];
//        for (int k = 0; k < numFilters; k++) {
//            for (int c = 0; c < filterDepth; c++) {
//                for (int i = 0; i < filterSize; i++) {
//                    for (int j = 0; j < filterSize; j++) {
//                        flipped_filters[k][c][i][j] = this.filters[k][c][filterSize - 1 - i][filterSize - 1 - j];
//                    }
//                }
//            }
//        }


//        // Pad the output gradient with zeros (assuming zero output_padding during forward pass)
//        double[][][] paddedOutput = new double[numFilters][paddedOutputHeight][paddedOutputWidth];
//        for (int k = 0; k < numFilters; k++) {
//            for (int h = 0; h < outputHeight; h++) {
//                for (int w = 0; w < outputWidth; w++) {
//                    paddedOutput[k][h + output_padding][w + output_padding] = outputGradient[k][h][w];
//                }
//            }
//        }

        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];

        /*
        * output_width = (int) Math.floor((double) (input_width - filterWidth + 2p) / x_stride + 1);
        *
        *
        * */
//        for(int d=0;d<inputDepth;d++){
//            inputGradient[d] = Convolution.convolve3d(outputGradient[d], flipped_filters[d], 1, stride, stride, 0, 0, 0)[0];
//            System.out.println("Shape : " + inputGradient[d].length + " " + inputGradient[d][0].length);
//            System.out.println("Supposed to be of shape : " + inputHeight + "x" + inputWidth);
//        }

        for (int d = 0; d < inputDepth; d++) {
            // we know that convolution of outputGradient with one filter will return 1xoutputHeightxoutputWidth, so we reshape it to outputHeightxoutputWidth
//            inputGradient[filter_idx] = Convolution.convolve3d(outputGradient, flipped_filters[filter_idx], stride, 0)[0];
            for (int inputY = 0; inputY < inputHeight; inputY++) {
                for (int inputX = 0; inputX < inputWidth; inputX++) {

                    double a = 0.0;
                    for (int k = 0; k < numFilters; k++) {

                        for (int fy = 0; fy < filterSize; fy++) {
                            for (int fx = 0; fx < filterSize; fx++) {
                                // Consider output_padding by using valid indices within padded output
                                int outputY = inputY + fy * stride;
                                int outputX = inputX + fx * stride;
                                int effectiveOutputY = outputY + output_padding;
                                int effectiveOutputX = outputX + output_padding;

                                // calculating the transposed filter indices
                                int fx_t = filterSize - 1 - fx;
                                int fy_t = filterSize - 1 - fy;

                                if ((0 <= effectiveOutputY && effectiveOutputY < outputHeight)
                                        && (0 <= effectiveOutputX && effectiveOutputX < outputWidth)) {
                                    a += this.filters[k][d][fy_t][fx_t] * outputGradient[k][effectiveOutputY][effectiveOutputX];
                                }
                            }
                        }
                    }
                    inputGradient[d][inputY][inputX] = a;
                }
            }
        }

//        System.out.println("paddedOutput");
//        UTIL.prettyprint(paddedOutput[0]);
//        System.out.println("InputGradient");
//        UTIL.prettyprint(inputGradient[0]);
//        System.out.println("Filter");
//        UTIL.prettyprint(filters[0][0]);

//        Convolution.test(paddedOutput, new double[][][]{flipped_filter});

        return inputGradient;
    }


    public void updateParameters(double[][][] outputGradient, double[][][] input, double learningRate) {
        /** calculates the filter gradients and bias gradients and updates the filters and biases

         apparently, filterGradients = convolve(input, outputGradient[k]) and biasGradients = sum(outputGradient[k])

         so that the output of convolution is of the same shape as the filterGradients, we have to pad the input accordingly

         in convolution : output_width = (int) Math.floor((double) (input_width - filterWidth + 2p) / x_stride + 1);
         but we want output dimension of convolution to be equal to filter dimensions, and the filter being used here will be the outputGradient
         substituting : filter_width = (int) Math.floor((double) (input_width - outputGradientWidth + 2p) / x_stride + 1);
         (stride * (filterSize - 1) + outputGradient_width - input_width)/2 = p;*/

        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];
        double[] biasGradient = new double[numFilters]; // to store bias gradients


        int hp = (int) Math.ceil((stride * (filterSize - 1) + outputHeight - inputHeight)/2.0);
        int wp = (int) Math.ceil((stride * (filterSize - 1) + outputWidth - inputWidth)/2.0);
        int dp = 0;//(int) Math.ceil((1 * (filterDepth - 1) + outputDepth - inputDepth)/2.0);// z_stride = 1
        System.out.println("hp : " + hp + " wp : " + wp + " dp : " + dp);

        for (int k = 0; k < numFilters; k++) {
            filtersGradient[k] = Convolution.convolve3d(Convolution.pad3d(input, dp, hp, wp), new double[][][]{outputGradient[k]}, 1, stride, stride, 0, 0, 0);
//            UTIL.prettyprint(filtersGradient[k]);
            biasGradient[k] = UTIL.sum(outputGradient[k]) / (outputGradient[k].length * outputGradient[k][0].length);
        }
        System.out.println("Shape : " + filtersGradient[0].length + " " + filtersGradient[0][0].length + " " + filtersGradient[0][0][0].length);
        System.out.println("Supposed to be of shape : " + filterDepth + "x" + filterSize + "x" + filterSize);
//
//        // Calculate filter gradients (same as before)
//        System.out.println("c : " + inputDepth);
//        for (int k = 0; k < numFilters; k++) {
//            for (int c = 0; c < filterDepth; c++) {
//                for (int i = 0; i < filterSize; i++) {
//                    for (int j = 0; j < filterSize; j++) {
//
//                        for (int h = 0; h < outputHeight; h++) {
//                            for (int w = 0; w < outputWidth; w++) {
//                                int inH = h + i * this.stride;
//                                int inW = w + j * this.stride;
//                                int effectiveInH = inH; // Consider padding
//                                int effectiveInW = inW;
//
//                                if ((0 <= effectiveInH && effectiveInH < inputHeight)
//                                        && (0 <= effectiveInW && effectiveInW < inputWidth)) {
//                                    filtersGradient[k][c][i][j] += outputGradient[k][h][w] * input[c][effectiveInH][effectiveInW];
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//        System.out.println("outputGradient");
//        UTIL.prettyprint(outputGradient[0]);
//        System.out.println("input");
//        UTIL.prettyprint(input[0]);
//        System.out.println("filterGradient");
//        UTIL.prettyprint(filtersGradient[0][0]);
//
//
//        // Calculate bias gradients
//        for (int k = 0; k < numFilters; k++) {
//            for (int h = 0; h < outputHeight; h++) {
//                for (int w = 0; w < outputWidth; w++) {
//                    biasGradient[k] += outputGradient[k][h][w]; // Sum over output
//                }
//            }
//            biasGradient[k] /= (outputWidth * outputHeight); // Average gradient
//        }

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
        updateParameters(outputGradient, this.input, learningRate);
    }


    public static void main(String[] args) {

        // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding

        TransposeConvolutionalLayer tconv2 = new TransposeConvolutionalLayer(2, 1, 1, 2, 2, 1, 0);

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

        double[][] targetOutput = new double[tconv2.outputHeight][tconv2.outputWidth];

        double[][][] input = {{{1.0, 1.0}, {1.0, 1.0}}};

        double learning_rate = 0.0005;

        for (int epoch = 0; epoch < 5001; epoch++) {

            //forward pass
            double[][][] tconv2_output = tconv2.forward(input);

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
            tconv2.updateParameters(new double[][][]{outputGradient}, learning_rate);

            System.out.println("Sum of values in each gradient :");
            System.out.println("Tconv2 input gradient: " + Arrays.stream(UTIL.flatten(tconv2_in_gradient_tconv_out_gradient[0])).sum());


            if (epoch % 5 == 0) {
                double[][] output = tconv2_output[0];

                double[][] filter = tconv2.filters[0][0];
                System.out.println("Filters:");
                UTIL.prettyprint(filter);

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