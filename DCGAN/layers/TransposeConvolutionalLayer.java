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
    public int input_padding = 0;
    public int output_padding = 0; // for now we don't care about output_padding

    boolean useBias = true;

    public TransposeConvolutionalLayer(int inputDepth, int filterSize, int numFilters, int stride) {
        this(filterSize, numFilters, stride, 28, 28, inputDepth, 0, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int input_padding) {
        this(filterSize, numFilters, stride, inputWidth, inputHeight, inputDepth, input_padding, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int input_padding, boolean useBias) {
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

        this.input_padding = input_padding;

        // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding

        outputHeight = this.stride * (inputHeight - 1) + filterSize - 2 * input_padding;
        outputWidth = this.stride * (inputWidth - 1) + filterSize - 2 * input_padding;
        outputDepth = numFilters;
    }

    double[][][] output;

    public double[][][] forward(double[][][] input) {
        this.input = input;

        double[][][] padded_input = Convolution.pad3d(input, 0, input_padding, input_padding);

        /* Preparing the input for transpose convolution */
        /* step 1 : insert zeroes of thickness equal to stride-1*/
        double[][][] stretched_tconv_input = UTIL.addZeroesInBetween(padded_input, 0, stride - 1, stride - 1);
        /* step 2 : pad the stretched_input with zeroes of thickness equal to filterSize - 1*/
        double[][][] padded_and_stretched_tconv_input = Convolution.pad3d(stretched_tconv_input, 0, filterSize - 1, filterSize - 1);

        double[][][] output = new double[numFilters][outputHeight][outputWidth];
        this.output = output;

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            // repeat the filter across the depth of the input (Why? Because we have 2d filters and 3d input and we want a 2d activation maps which we stack to get a 3d output)
            double[][][] repeated_filter = new double[inputDepth][filterSize][filterSize];
            for (int c = 0; c < inputDepth; c++) {
                repeated_filter[c] = filters[filter_idx];
            }
            /*
                [[1,1]
                [1,1]]
            * and input depth is 3, then filter is treated as
            *  [[1,1]    [[1,1]     [[1,1]
                [1,1]]   [1,1]]      [1,1]
             * */

            output[filter_idx] = Convolution.convolve3d(padded_and_stretched_tconv_input, repeated_filter, 1, 1, 1, 0, 0, 0)[0];
        } // each activation map : 1 x outputHeigh x outputWidth reshaped to ---> outputHeight x outputWidth
        // stack the activation maps to get the output

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

        double[][] inputGradientSlice = Convolution.convolve3d(outputGradient, flipped_filters, 1, stride, stride, dp, hp, wp)[0]; // new double[inputDepth][inputHeight][inputWidth];

        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];
        for (int c = 0; c < inputDepth; c++) {
            for (int slice_i = input_padding; slice_i < inputHeight + input_padding; slice_i++) {
                for (int slice_j = input_padding; slice_j < inputWidth + input_padding; slice_j++) {
                    inputGradient[c][slice_i - input_padding][slice_j - input_padding] = inputGradientSlice[slice_i][slice_j];
                }
            }
        }

        if (inputHeight != inputGradientSlice.length - 2 * input_padding || inputWidth != inputGradientSlice[0].length - 2 * input_padding) {
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
        int hp = 0; // (int) Math.ceil((stride * (filterSize - 1) - outputHeight + (inputHeight)) / 2.0);
        int wp = 0; // (int) Math.ceil((stride * (filterSize - 1) - outputWidth + (inputWidth)) / 2.0);
        int dp = 0; // (int) Math.floor((1 * (1 - 1) + inputDepth  - outputDepth) / 2.0);// z_stride = 1

        // output_shape = ((input_height - kernel_size + 2 * padding) / stride) + 1

        System.out.println("for dou J/dou W hp : " + hp + " wp : " + wp + " dp : " + dp);

        System.out.println("outputgradient shape : " + outputGradient.length + " " + outputGradient[0].length + " " + outputGradient[0][0].length);
        System.out.println("input shape : " + input.length + " " + input[0].length + " " + input[0][0].length);
        System.out.println("output shape according to formula : " + outputDepth + " " + outputHeight + " " + outputWidth);
        System.out.println("Actual output shape : " + output.length + " " + output[0].length + " " + output[0][0].length);
        System.out.println("flipped filters shape : " + filters[0].length + " " + filters[0][0].length);
        System.out.println("stride : " + stride);
        System.out.println("Stretched and padded input : ");

        // for convolution : output_shape = ((input_height - kernel_size -2*kp + 2 * padding) / stride) + 1
        // so to get the same shape as the filter, we have to increase the stride
        // filter_shape = (output_shape - input_shape - 2*ip)/stride + 1
        double[][][] paddedInput = Convolution.pad3d(input, 0, input_padding, input_padding);

        for (int k = 0; k < numFilters; k++) {
            double[][][] reapeatedOutputSlice = new double[inputDepth][][];

            double[][] rotatedOutputGradientSlice = UTIL.rotate180(outputGradient[k]);
            for (int c = 0; c < inputDepth; c++) {
                reapeatedOutputSlice[c] = rotatedOutputGradientSlice;
            }

            filtersGradient[k] = Convolution.convolve3d(reapeatedOutputSlice, paddedInput, 1, stride, stride, dp, hp, wp)[0];
//            UTIL.prettyprint(filtersGradient[k]);
            biasGradient[k] = UTIL.sum(outputGradient[k]) / (outputGradient[k].length * outputGradient[k][0].length);
        }

        if (filtersGradient[0].length != filterSize || filtersGradient[0][0].length != filterSize) {
            // warning
            System.out.println("Warning : filtersGradient shape is not as expected. Please change previous or next or current layer dimensions to avoid errors.");
            System.out.println("hp : " + hp + " wp : " + wp + " dp : " + dp);

            System.out.println("each filterGradient shape : " + filtersGradient[0].length + " " + filtersGradient[0][0].length);
            System.out.println("Supposed to be of shape : " + 1 + "x" + filterSize + "x" + filterSize);
            System.out.println("hp : " + hp + " wp : " + wp + " dp : " + dp);
            System.out.println("filterDepth : " + 1 + " outputDepth : " + outputDepth + " inputDepth : " + inputDepth + " Math.floor((1 * (filterDepth - 1) + outputDepth - inputDepth) / 2.0)");
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
        double[][][] input = new double[1][24][24]; // XavierInitializer.xavierInit3D(1, 24, 24);

        for(int i = 0; i < input[0].length; i++) {
            for(int j = 0; j < input[0][0].length; j++) {
                input[0][i][j] = 1; // i + j;
            }
        }

        TransposeConvolutionalLayer layer = new TransposeConvolutionalLayer(
                3, 1, 1,
                input[0][0].length, input[0].length, 1,
                0, false);

        double[][] filter = {
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        };
        layer.filters[0] = filter;

        double[] biases = new double[1];
        layer.biases = biases;

        double[][][] output = layer.forward(input);
        output = UTIL.multiplyScalar(output, 1.0 / 9.0);

        UTIL.saveImage(UTIL.getBufferedImage(output[0]), "tconv_output.png");

        System.out.println("Output:");
        for (int i = 0; i < output[0].length; i++) {
            for (int j = 0; j < output[0][0].length; j++) {
                System.out.print(output[0][i][j] + " ");
            }
            System.out.println();
        }
        System.exit(0);

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
//         When we want it to learn the 1s only matrix filter
        double[][] targetOutput = new double[layer.outputHeight][layer.outputWidth];
        for (int i = 0; i < targetOutput.length; i++) {
            for (int j = 0; j < targetOutput[0].length; j++) {
                targetOutput[i][j] = 1;
            }
        }

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
        for (int epoch = 0; epoch < 10000001 && (loss > 0.001) && !Double.isNaN(loss) && !Double.isInfinite(loss); epoch++) {

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