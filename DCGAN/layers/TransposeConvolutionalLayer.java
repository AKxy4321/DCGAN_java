package DCGAN.layers;

import DCGAN.UTIL;
import DCGAN.XavierInitializer;

public class TransposeConvolutionalLayer {
    double[][][] filters;
    double[] biases;
    private final int stride;
    double[][][] input;
    public int numFilters;

    public int filterSize;
    public int inputWidth, inputHeight, inputDepth;
    public int outputWidth, outputHeight, outputDepth;
    public int padding = 0;
    public int output_padding = 0; // for now we don't care about output_padding

    boolean useBias = true;

    public TransposeConvolutionalLayer(int inputDepth, int filterSize, int numFilters, int stride) {
        this(filterSize, numFilters, stride, 28, 28, inputDepth, 0, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding) {
        this(filterSize, numFilters, stride, inputWidth, inputHeight, inputDepth, padding, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding, boolean useBias) {
        this.useBias = useBias;
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filters = new double[numFilters][filterSize][filterSize];
        this.biases = new double[numFilters];

        this.filters = XavierInitializer.xavierInit3D(numFilters, filterSize, filterSize);
        this.biases = XavierInitializer.xavierInit1D(numFilters);
        this.stride = stride;

        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        this.padding = padding;

        // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding

        outputHeight = this.stride * (inputHeight - 1) + filterSize - 2 * padding;
        outputWidth = this.stride * (inputWidth - 1) + filterSize - 2 * padding;
        outputDepth = numFilters;
    }

    public double[][][] forward(double[][][] input) {
        this.input = input;
        int paddedInputHeight = inputHeight + padding * 2;
        int paddedInputWidth = inputWidth + padding * 2;

        int filterDepth = 1;

        // numFilters = (int) Math.floor((double) (stretched_input_depth - filterDepth + 2p) / z_stride + 1);
        // numFilters = (1 - 1 + 2 * p)/1 + 1;
        // numFilters = 2p + 1
        // p = (numFilters - 1)/2

        // Pad the input with zeros
//        double[][][] paddedInput = Convolution.pad3d(input, 0, filterSize-1, filterSize-1);

//        double[][][] stretched_input = UTIL.addZeroesInBetween(paddedInput, 0, stride-1,stride-1);
//        System.out.println("Padded Input : ");
//        UTIL.prettyprint(paddedInput);

//        System.out.println("Filters : ");
//        UTIL.prettyprint(filters);

//        double[][][] output = new double[numFilters][outputHeight][outputWidth];
//
//        for(int filter_idx = 0;filter_idx<numFilters;filter_idx++){
//            double[][][] repeated_filter = new double[inputDepth][filterSize][filterSize];
//            for(int c = 0; c < inputDepth; c++) {
//                repeated_filter[c] = filters[filter_idx];
//            }
//            output[filter_idx] = Convolution.convolve3d(stretched_input, repeated_filter, 1, 1, 1, 0, 0,0)[0];
//        }
//        output = Convolution.convolve3d(stretched_input, filters, 1, 1, 1, 0, 0,0);


        double[][][] output = new double[numFilters][outputHeight][outputWidth];

        for (int oy = 0; oy < outputHeight; oy++) {
            for (int ox = 0; ox < outputWidth; ox++) {
                for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {

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
                                    sum += input[c][effectiveInH][effectiveInW] * this.filters[filter_idx][fy][fx];
                                }
                            }
                        }
                    }

                    output[filter_idx][oy][ox] = sum + (useBias ? this.biases[filter_idx] : 0); // Add bias to the dot product
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
         * with W^T and W, respectively.
         * */


        /*
         * the formula to calculate the dimension of output for convolution is
         * output_width = (int) Math.floor((double) (input_width - filterWidth + 2p) / x_stride + 1);
         *
         * we have to pad so that convolution result has the same dimensions as 1xinput_heightxinput_width
         * substitute the values
         * input = (output_gradient - filter +2p)/stride + 1
         * ((input - 1)*stride + filter - outputGradient)/2 = p
         *
         * */

        double[][][] flipped_filters = new double[numFilters][filterSize][filterSize];
        for (int k = 0; k < numFilters; k++) {
            flipped_filters[k] = UTIL.rotate180(filters[k]);
        }

        // calculating padding
        int dp = 0; // (int) Math.ceil((1 * (inputDepth - 1) + 1 - outputDepth) / 2.0); // z_stride = 1
        int hp = (int) Math.ceil((stride * (inputHeight - 1) + filterSize - outputHeight) / 2.0);
        int wp = (int) Math.ceil((stride * (inputWidth - 1) + filterSize - outputWidth) / 2.0);

        System.out.printf("for dou J/dou X hp : %d wp : %d dp : %d\n", hp, wp, dp);

        double[][] inputGradientSlice = Convolution.convolve3d(outputGradient, filters, 1, stride, stride, dp, hp, wp)[0]; // new double[inputDepth][inputHeight][inputWidth];

        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];
        for (int c = 0; c < inputDepth; c++) {
            inputGradient[c] = inputGradientSlice;
        }
//
        // Sum gradients across channels
//        for(int c = 0; c < inputDepth; c++) {
//            for(int i = 0; i < inputHeight; i++) {
//                for(int j = 0; j < inputWidth; j++) {
//                    for(int k = 0; k < outputDepth; k++) {
//                        for(int u = 0; u < filterSize; u++) {
//                            for(int v = 0; v < filterSize; v++) {
//                                int di = i - u;
//                                int dj = j - v;
//                                if (di >= 0 && di < outputHeight && dj >= 0 && dj < outputWidth) {
//                                    inputGradient[c][i][j] += outputGradient[k][di][dj] * filters[k][u][v];
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }


        if (inputDepth != inputGradient.length || inputWidth != inputGradient[0].length || inputWidth != inputGradient[0][0].length) {
            // warning
            System.out.println("Warning : inputGradient shape is not as expected. Please change layer dimensions to avoid errors.");
            System.out.println("hp : " + hp + " wp : " + wp + " dp : " + dp);
            System.out.println("dp supposed to be : " + (1 * (inputDepth - 1) + 1 - outputDepth) / 2.0);

            System.out.println("filterSize : " + filterSize + " filterDepth : " + 1);
            System.out.println("inputDepth : " + inputDepth + " inputHeight : " + inputHeight + " inputWidth : " + inputWidth);
            System.out.println("outputDepth : " + outputDepth + " outputHeight : " + outputHeight + " outputWidth : " + outputWidth);
            System.out.println("Update Parameters method:");
            System.out.println("inputGradientPerFilter is supposed to be of shape : " + inputDepth + "x" + inputHeight + "x" + inputWidth);
            System.out.println("inputGradient Shape : " + inputGradient.length + " " + inputGradient[0].length + " " + inputGradient[0][0].length);
        }

        return inputGradient;
    }


    public void updateParameters(double[][][] outputGradient, double[][][] input, double learningRate) {
        /** calculates the filter gradients and bias gradients and updates the filters and biases

         apparently, filterGradients = convolve(input, 180 rotated outputGradient[k]) and biasGradients = sum(outputGradient[k])

         so that the output of convolution is of the same shape as the filterGradients, we have to pad the input accordingly

         in convolution : output_shape = ((input_height - kernel_size + 2 * padding) / stride) + 1
         but we want output dimension of convolution to be equal to filter dimensions, and the filter being used here will be the outputGradient
         substituting : filter_width = (int) Math.floor((double) (input_width - outputGradientWidth + 2p) / x_stride + 1);
         (stride * (filterSize - 1) + outputGradient_width - input_width)/2 = p;*/

        double[][][] filtersGradient = new double[numFilters][filterSize][filterSize];
        double[] biasGradient = new double[numFilters]; // to store bias gradients

        // calculating the padding
        int hp = (int) Math.ceil((stride * (filterSize - 1) - outputHeight + inputHeight) / 2.0);
        int wp = (int) Math.ceil((stride * (filterSize - 1) - outputWidth + inputWidth) / 2.0);
        int dp = 0; // (int) Math.floor((1 * (1 - 1) + inputDepth  - outputDepth) / 2.0);// z_stride = 1

        // output_shape = ((input_height - kernel_size + 2 * padding) / stride) + 1

        double[][][] rotatedInput = UTIL.rotate180(input);
//        double[][][] transposedInput = UTIL.transpose(input);

        for (int k = 0; k < numFilters; k++) {
            double[][][] reapeatedOutputSlice = new double[inputDepth][][];

            double[][] rotatedOutputGradientSlice = outputGradient[k];
            for (int c = 0; c < inputDepth; c++) {
                reapeatedOutputSlice[c] = rotatedOutputGradientSlice;
            }

            filtersGradient[k] = Convolution.convolve3d(reapeatedOutputSlice, rotatedInput, 1, stride, stride, dp, hp, wp)[0];
//            UTIL.prettyprint(filtersGradient[k]);
            biasGradient[k] = UTIL.sum(outputGradient[k]) / (outputGradient[k].length * outputGradient[k][0].length);
        }

        if (filtersGradient[0].length != filterSize || filtersGradient[0][0].length != filterSize) {
            // warning
            System.err.println("Warning : filtersGradient shape is not as expected. Please change previous or next or current layer dimensions to avoid errors.");
            System.out.println("hp : " + hp + " wp : " + wp + " dp : " + dp);

            System.err.println("each filterGradient shape : " + filtersGradient[0].length + " " + filtersGradient[0][0].length);
            System.err.println("Supposed to be of shape : " + 1 + "x" + filterSize + "x" + filterSize);
            System.err.println("hp : " + hp + " wp : " + wp + " dp : " + dp);
            System.err.println("filterDepth : " + 1 + " outputDepth : " + outputDepth + " inputDepth : " + inputDepth + " Math.floor((1 * (filterDepth - 1) + outputDepth - inputDepth) / 2.0)");
        }

        // Update filters and biases
        for (int k = 0; k < numFilters; k++) {
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    this.filters[k][i][j] -= learningRate * filtersGradient[k][i][j];
                }
            }
            this.biases[k] -= learningRate * biasGradient[k];
        }
    }


    public void updateParameters(double[][][] outputGradient, double learningRate) {
        updateParameters(outputGradient, this.input, learningRate);
    }


    public static void main(String[] args) {
        double[][][] input = {{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1},
        }};
        TransposeConvolutionalLayer layer = new TransposeConvolutionalLayer(2, 1, 1, input[0][0].length, input[0].length, 1, 0, false);

        double[][] filter = {
                {2, 1},
                {4, 3}
        };
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

//        // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding
//
//        TransposeConvolutionalLayer tconv2 = new TransposeConvolutionalLayer(2, 1, 1, 2, 2, 1, 0, false);
//
////        double[][] targetOutput = {
////                {0, 0, 0, 0},
////                {0, 3, 2, 3},
////                {0, 2, 0, 3},
////                {4, 6, 6, 9}
////        };
//
////        double[][] targetOutput = {
////                {0, 0, 0, 0},
////                {0, 0, 0, 0},
////                {0, 0, 0, 0},
////                {0, 0, 0, 0},
////        };
//
        // When we want it to learn the null matrix filter, we give a null matrix target output
        // double[][] targetOutput = new double[layer.outputHeight][layer.outputWidth];

        // when we want it to learn the filter [[1,2],[3,4]]
        input = new double[][][]{{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1},
        }};
        layer = new TransposeConvolutionalLayer(2, 1, 1, input[0][0].length, input[0].length, 1, 0, false);

        layer.filters[0] = new double[][]{
                {1, 1},
                {0, 0}
        };

//        double[][] targetOutput = {
//                {1.0, 3.0, 3.0, 2.0},
//                {4.0, 10.0, 10.0, 6.0},
//                {4.0, 10.0, 10.0, 6.0},
//                {3.0, 7.0, 7.0, 4.0},
//        };

        // when we want it to learn the filter {{2, 1},{4, 3}}
        double[][] targetOutput = {
                {2.0, 3.0, 3.0, 1.0},
                {6.0, 10.0, 10.0, 4.0},
                {6.0, 10.0, 10.0, 4.0},
                {4.0, 7.0, 7.0, 3.0},
        };

//        targetOutput = UTIL.multiplyScalar(targetOutput,0);

        // When we want it to learn the filter [[1,0],[0,1]]
//        double[][] targetOutput = {
//                {1, 1, 0},
//                {1, 2, 1},
//                {0, 1, 1}};
//
//
////         When we want it to learn the filter [[1,1],[1,1]]
////        double[][] targetOutput = {
////                {1, 2, 1},
////                {2, 4, 2},
////                {1, 2, 1}};
//
        double learning_rate = 0.01;
        double loss = Double.MAX_VALUE;
        for (int epoch = 0; epoch < 100000001 && (loss > 0.000001) && !Double.isNaN(loss) && !Double.isInfinite(loss); epoch++) {

            //forward pass
            double[][][] tconv2_output = layer.forward(input);

            double[][] outputGradient = new double[tconv2_output[0].length][tconv2_output[0][0].length];
            UTIL.calculateGradientMSE(outputGradient, tconv2_output[0], targetOutput);
            loss = UTIL.lossMSE(tconv2_output[0], targetOutput);


            System.out.println("Epoch " + (epoch + 1) + ", loss: " + loss);

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
            double[][][] tconv2_in_gradient_tconv_out_gradient = layer.backward(new double[][][]{outputGradient});

            layer.updateParameters(new double[][][]{outputGradient}, learning_rate);

//            System.out.println("Sum of values in each gradient :");
//            System.out.println("Tconv2 input gradient: " + Arrays.stream(UTIL.flatten(tconv2_in_gradient_tconv_out_gradient[0])).sum());


            if (epoch % 5 == 0) {
                output = tconv2_output;

                filter = layer.filters[0];
//                System.out.println("Filters:");
                UTIL.prettyprint(filter);

                System.out.println("Target Output:");
                UTIL.prettyprint(targetOutput);

                System.out.println("Output:");
                UTIL.prettyprint(output);

                System.out.println("tconv2_in_gradient_tconv_out_gradient shape : "
                        + tconv2_in_gradient_tconv_out_gradient.length + " "
                        + tconv2_in_gradient_tconv_out_gradient[0].length + " "
                        + tconv2_in_gradient_tconv_out_gradient[0][0].length);
            }
        }

        UTIL.prettyprint(layer.filters[0]);

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