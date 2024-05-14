package DCGAN.layers;

import DCGAN.util.MiscUtils;
import DCGAN.util.XavierInitializer;

import static DCGAN.util.TrainingUtils.calculateGradientRMSE;
import static DCGAN.util.TrainingUtils.lossMSE;

public class TransposeConvolutionalLayer {
    double[][][][] filters;
    double[] biases;
    private final int stride;
    double[][][] input;
    public int numFilters;

    public int filterSize, filterDepth;
    public int inputWidth, inputHeight, inputDepth;
    public int outputWidth, outputHeight, outputDepth;
    public int padding = 0, right_bottom_padding = 0;
    public int output_padding = 0; // for now we don't care about output_padding

    boolean useBias = true;

    public TransposeConvolutionalLayer(int inputDepth, int filterSize, int numFilters, int stride) {
        this(filterSize, numFilters, stride, 28, 28, inputDepth, 0,0,0, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding) {
        this(filterSize, numFilters, stride, inputWidth, inputHeight, inputDepth, padding,0,0, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding, boolean useBias){
        this(filterSize, numFilters, stride, inputWidth, inputHeight, inputDepth, padding,0,0, useBias);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding, int output_padding, int right_bottom_padding, boolean useBias) {
        this.useBias = useBias;
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filterDepth = inputDepth;
        this.filters = new double[numFilters][filterDepth][filterSize][filterSize];
        this.biases = new double[numFilters];

        this.filters = XavierInitializer.xavierInit4D(numFilters, filterDepth, filterSize);
        this.biases = XavierInitializer.xavierInit1D(numFilters);
        this.stride = stride;

        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        this.padding = padding;
        this.output_padding = output_padding;
        this.right_bottom_padding = right_bottom_padding;

        // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding

        outputHeight = this.stride * (inputHeight - 1) + filterSize - 2 * padding + output_padding + right_bottom_padding;
        outputWidth = this.stride * (inputWidth - 1) + filterSize - 2 * padding + output_padding + right_bottom_padding;

        outputDepth = numFilters;
    }

    public double[][][] forward(double[][][] input) {
        // considering "same" padding
        this.input = input;

        double[][][][] rotated_filters = new double[numFilters][filterDepth][filterSize][filterSize];
        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            for (int fd = 0; fd < filterDepth; fd++) {
                rotated_filters[filter_idx][fd] = MiscUtils.rotate90(MiscUtils.rotate90(filters[filter_idx][fd]));
            }
        }

        double[][][] stretched_input = MiscUtils.addZeroesInBetween(input, 0, stride - 1, stride - 1);
        // "same" padding
        double[][][] padded_and_stretched_input = Convolution.pad3d(stretched_input, 0, 0, padding, padding + right_bottom_padding, padding, padding + right_bottom_padding);

        double[][][] output = new double[numFilters][outputHeight][outputWidth];

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            output[filter_idx] = Convolution.convolve3d(padded_and_stretched_input, rotated_filters[filter_idx], 1, 1, 1)[0];
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

        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            double[][][] inputGradientSlice = Convolution.convolve3d(
                    Convolution.pad3d(
                            new double[][][]{outputGradient[filter_idx]},
                            filterDepth - 1, filterDepth - 1,
                            padding, padding,
                            padding, padding),
                    filters[filter_idx],
                    1, stride, stride);

            if(inputGradientSlice[0].length != inputHeight){
                System.out.println("Input gradient slice shape not as expected");
            }

            for (int c = 0; c < inputDepth; c++) {
                for (int i = 0; i < inputHeight; i++) {
                    for (int j = 0; j < inputWidth; j++) {
                        inputGradient[inputDepth - 1 - c][i][j] += inputGradientSlice[c][i][j];
                    }
                }
            }
        }

        if (inputDepth != inputGradient.length || inputWidth != inputGradient[0].length || inputWidth != inputGradient[0][0].length) {
            // warning
            System.out.println("Warning : inputGradient shape is not as expected. Please change layer dimensions to avoid errors.");
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

        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];
        double[] biasGradient = new double[numFilters]; // to store bias gradients

        double[][][] stretched_input = MiscUtils.addZeroesInBetween(input, 0, stride - 1, stride - 1);

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            double[][][] result = Convolution.convolve3d(
                    Convolution.pad3d(new double[][][]{outputGradient[filter_idx]},
                            filterDepth - 1, filterDepth - 1,
                            padding, padding - right_bottom_padding,
                            padding, padding - right_bottom_padding
                    ),
                    stretched_input,
                    1, 1, 1);

//            System.out.println("outputGradient[" + filter_idx + "]");
//            UTIL.prettyprint(outputGradient[filter_idx]);
//
//            System.out.println("input");
//            UTIL.prettyprint(input);
//
//            System.out.println("result");
//            UTIL.prettyprint(result);
//            System.exit(0);

            double[][][] reversed_along_depth = new double[result.length][result[0].length][result[0][0].length];
            for (int d = 0; d < result.length; d++) {
                for (int i = 0; i < result[0].length; i++) {
                    for (int j = 0; j < result[0][0].length; j++) {
                        reversed_along_depth[result.length - 1 - d][i][j] = result[d][i][j];
                    }
                }
            }
            result = reversed_along_depth;

            filtersGradient[filter_idx] = result;
        }

        if (filtersGradient[0].length != filterDepth || filtersGradient[0][0].length != filterSize || filtersGradient[0][0][0].length != filterSize) {
            // warning
            System.err.println("Warning : filtersGradient shape is not as expected. Please change previous or next or current layer dimensions to avoid errors.");
//            System.out.println("hp : " + hp + " wp : " + wp + " dp : " + dp);

            System.err.println("each filterGradient shape : " + filtersGradient[0].length + " " + filtersGradient[0][0].length + " "+ filtersGradient[0][0][0].length);
            System.err.println("Supposed to be of shape : " + 1 + "x" + filterSize + "x" + filterSize);
//            System.err.println("hp : " + hp + " wp : " + wp + " dp : " + dp);
            System.err.println("filterDepth : " + 1 + " outputDepth : " + outputDepth + " inputDepth : " + inputDepth + " Math.floor((1 * (filterDepth - 1) + outputDepth - inputDepth) / 2.0)");
        }

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            for (int fd = 0; fd < filterDepth; fd++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    for (int fx = 0; fx < filterSize; fx++) {
                        filters[filter_idx][fd][fy][fx] -= learningRate * filtersGradient[filter_idx][fd][fy][fx];
                    }
                }
            }
        }

    }


    public void updateParameters(double[][][] outputGradient, double learningRate) {
        updateParameters(outputGradient, this.input, learningRate);
    }


    public static void main(String[] args) {
        double[][][] input = {
                {
                        {1, 2},
                        {3, 4}
                },
                {
                        {1, 2},
                        {3, 4}
                },
                {
                        {1, 2},
                        {3, 4}
                }
        };
        System.out.println("Actual input : ");
        MiscUtils.prettyprint(input);

        TransposeConvolutionalLayer layer = new TransposeConvolutionalLayer(2, 1, 1, input[0][0].length, input[0].length, input.length, 0, false);

        double[][][] filter = {
                {
                        {1, 2},
                        {3, 4}
                },
                {
                        {1, 2},
                        {3, 4}
                },
                {
                        {1, 2},
                        {3, 4}
                }};
        layer.filters[0] = filter;
        System.out.println("Filters length : " + layer.filters.length);
//        layer.filters[1] = filter.clone();

        double[] biases = new double[1];
        layer.biases = biases;

        double[][][] output = layer.forward(input);
        System.out.println("Output:");
        MiscUtils.prettyprint(output);

        double[][][] targetOutput = new double[][][]{{
                {3.0, 12.0, 12.0},
                {18.0, 60.0, 48.0},
                {27.0, 72.0, 48.0}}
        };
        layer = new TransposeConvolutionalLayer(2, 1, 1, input[0][0].length, input[0].length, input.length, 0, false);


//        System.exit(0);

//        double[][][] input = {{
//                {1, 2},
//                {3, 4}
//        }};
//        TransposeConvolutionalLayer layer = new TransposeConvolutionalLayer(2, 1, 1, input[0][0].length, input[0].length, 1, 0, false);
//
////        double[][] filter = {
////                {2, 1},
////                {4, 3}
////        };
//
//        double[][] filter = {
//                {1, 2},
//                {3, 4}
//        };
//        layer.filters[0][0] = filter;
//
//        double[] biases = new double[1];
//        layer.biases = biases;
//
//        double[][][] output = layer.forward(input);
//        System.out.println("Output:");
//        for (int i = 0; i < output[0].length; i++) {
//            for (int j = 0; j < output[0][0].length; j++) {
//                System.out.print(output[0][i][j] + " ");
//            }
//            System.out.println();
//        }
//        System.exit(0);

//        // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding

        // When we want it to learn the null matrix filter, we give a null matrix target output
        // double[][] targetOutput = new double[layer.outputHeight][layer.outputWidth];

        // when we want it to learn the filter [[1,2],[3,4]]
//        input = new double[][][]{{
//                {1, 1, 1},
//                {1, 1, 1},
//                {1, 1, 1},
//        }};


//        input = new double[][][]{{
//                {1, 0, 2},
//                {5, 1, 0},
//                {6, 3, 8},
//        }};
//        double[][][] targetOutput = new double[][][]{{
//                {2.0, 1.0, 4.0, 2.0},
//                {14.0, 10.0, 9.0, 6.0},
//                {32.0, 31.0, 22.0, 8.0},
//                {24.0, 30.0, 41.0, 24.0}
//        },
//                {
//                        {3.0, 4.0, 6.0, 8.0},
//                        {16.0, 25.0, 6.0, 4.0},
//                        {23.0, 44.0, 38.0, 32.0},
//                        {6.0, 15.0, 14.0, 16.0},
//                }
//        };
//        layer = new TransposeConvolutionalLayer(2, 2, 1, input[0][0].length, input[0].length, 1, 0, false);

//        layer.filters[0][0] = new double[][]{
//                {1, 1},
//                {1, 1}
//        };
//        layer.filters[1][0] = new double[][]{
//                {1, 1},
//                {1, 1}
//        };

//        double[][] targetOutput = {
//                {1.0, 3.0, 3.0, 2.0},
//                {4.0, 10.0, 10.0, 6.0},
//                {4.0, 10.0, 10.0, 6.0},
//                {3.0, 7.0, 7.0, 4.0},
//        };

        // when we want it to learn the filter {{2, 1},{4, 3}}
//        double[][] targetOutput = {
//                {2.0, 3.0, 3.0, 1.0},
//                {6.0, 10.0, 10.0, 4.0},
//                {6.0, 10.0, 10.0, 4.0},
//                {4.0, 7.0, 7.0, 3.0},
//        };

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
        double learning_rate = 0.001;
        double loss = Double.MAX_VALUE;
        for (int epoch = 0; epoch < 100000001 && (loss > 0.000001) && !Double.isNaN(loss) && !Double.isInfinite(loss); epoch++) {

            System.out.println("Filters:");
            for (int filter_idx = 0; filter_idx < layer.numFilters; filter_idx++)
                MiscUtils.prettyprint(layer.filters[filter_idx]);

            //forward pass
            double[][][] tconv2_output = layer.forward(input);

            double[][][] outputGradient = new double[tconv2_output.length][tconv2_output[0].length][tconv2_output[0][0].length];
            calculateGradientRMSE(outputGradient, tconv2_output, targetOutput);
            loss = lossMSE(tconv2_output, targetOutput);


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
            double[][][] tconv2_in_gradient_tconv_out_gradient = layer.backward(outputGradient);

            layer.updateParameters(outputGradient, learning_rate);

//            System.out.println("Sum of values in each gradient :");
//            System.out.println("Tconv2 input gradient: " + Arrays.stream(UTIL.flatten(tconv2_in_gradient_tconv_out_gradient[0])).sum());


            if (epoch % 5 == 0) {
                output = tconv2_output;

                System.out.println("Filters:");
                for (int filter_idx = 0; filter_idx < layer.numFilters; filter_idx++)
                    MiscUtils.prettyprint(layer.filters[filter_idx]);

                System.out.println("Target Output:");
                MiscUtils.prettyprint(targetOutput);

                System.out.println("Output:");
                MiscUtils.prettyprint(output);

                System.out.println("tconv2_in_gradient_tconv_out_gradient shape : "
                        + tconv2_in_gradient_tconv_out_gradient.length + " "
                        + tconv2_in_gradient_tconv_out_gradient[0].length + " "
                        + tconv2_in_gradient_tconv_out_gradient[0][0].length);
            }
        }

        MiscUtils.prettyprint(layer.filters[0]);

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