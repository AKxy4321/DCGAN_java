package DCGAN.layers;

import DCGAN.optimizers.AdamOptimizer;
import DCGAN.util.MiscUtils;
import DCGAN.util.XavierInitializer;

import java.util.Random;

import static DCGAN.layers.Convolution.pad3d;
import static DCGAN.util.MiscUtils.*;
import static DCGAN.util.TrainingUtils.calculateGradientRMSE;
import static DCGAN.util.TrainingUtils.lossMSE;

public class TransposeConvolutionalLayer {
    public double[][][][] filters;
    //    double[] biases;
    private final int stride;
    double[][][] input;
    public int numFilters;

    public int filterSize, filterDepth;
    public int inputWidth, inputHeight, inputDepth;
    public int outputWidth, outputHeight, outputDepth;
    public int padding = 0, right_bottom_padding = 0;
    public int output_padding = 0; // for now we don't care about output_padding

    int outputGradientPadding = 0;

    boolean useBias = false;

    AdamOptimizer filtersOptimizer;

    double[][][][] filtersGradient;
    double[][][] inputGradient;

    double[][][] stretched_input;
    double[][][] padded_and_stretched_input;
    double[][][] output;
    double[][][] output_slice;
    double[][][][] rotated_filters;
    double[][][] paddedOutputGradientSlice;
    double[][][] inputGradientSlice;


    public TransposeConvolutionalLayer(int inputDepth, int filterSize, int numFilters, int stride) {
        this(filterSize, numFilters, stride, 28, 28, inputDepth, 0, 0, 0, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding) {
        this(filterSize, numFilters, stride, inputWidth, inputHeight, inputDepth, padding, 0, 0, true);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding, boolean useBias) {
        this(filterSize, numFilters, stride, inputWidth, inputHeight, inputDepth, padding, 0, 0, useBias);
    }

    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding, int output_padding, int right_bottom_padding, boolean useBias) {
        this(filterSize, numFilters, stride, inputWidth, inputHeight, inputDepth, padding, output_padding, right_bottom_padding, 0, useBias, 0.001);
    }


    public TransposeConvolutionalLayer(int filterSize, int numFilters, int stride, int inputWidth, int inputHeight, int inputDepth, int padding, int output_padding, int right_bottom_padding, int outputGradientPadding, boolean useBias, double learning_rate) {
        this.useBias = useBias;

        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filterDepth = inputDepth;

        this.stride = stride;

        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        this.padding = padding;
        this.output_padding = output_padding;
        this.right_bottom_padding = right_bottom_padding;
        this.outputGradientPadding = outputGradientPadding;

//        this.biases = new double[numFilters]; // bias not implemented

        filters = new double[numFilters][filterDepth][filterSize][filterSize]; // XavierInitializer.xavierInit4D(numFilters, filterDepth, filterSize);
        Random random = new Random();
        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            filters[filter_idx] = XavierInitializer.xavierInit3D(filterDepth, filterSize, filterSize);
//            for(int fd = 0; fd < filterDepth; fd++) {
//                for(int i=0;i<filterSize;i++){
//                    for(int j=0;j<filterSize;j++){
//                        filters[filter_idx][fd][i][j] = random.nextGaussian(0,0.02);
//                    }
//                }
//            }
        }

        filtersOptimizer = new AdamOptimizer(numFilters * filterDepth * filterSize * filterSize, learning_rate, 0.5, 0.999, 1e-8);

        /**
         * Ok, so a major reason why the network training is slow is because we keep making too many new arrays in the forward and backward passes.
         * So, for efficiency's sake, we just initialize them once here, and reuse them in other places.
         * */
        filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];
        inputGradient = new double[inputDepth][inputHeight][inputWidth];
        stretched_input = new double[inputDepth + (inputDepth - 1) * (0)]
                [inputHeight + (inputHeight - 1) * (stride - 1)]
                [inputWidth + (inputWidth - 1) * (stride - 1)];
        int stretched_input_depth = stretched_input.length;
        int stretched_input_height = stretched_input[0].length;
        int stretched_input_width = stretched_input[0][0].length;
        int paddedInputDepth = stretched_input_depth + 0 + 0;
        int paddedInputHeight = stretched_input_height + padding + padding + right_bottom_padding;
        int paddedInputWidth = stretched_input_width + padding + padding + right_bottom_padding;
        padded_and_stretched_input = new double[paddedInputDepth][paddedInputHeight][paddedInputWidth];

        rotated_filters = new double[numFilters][filterDepth][filterSize][filterSize];

        this.outputWidth = (int) Math.floor((double) (paddedInputWidth - filterSize) / 1 + 1);
        this.outputHeight = (int) Math.floor((double) (paddedInputHeight - filterSize) / 1 + 1);
        this.outputDepth = numFilters;

        paddedOutputGradientSlice = new double[1 + 2 * (filterDepth - 1)][outputHeight + 2 * outputGradientPadding][outputWidth + 2 * outputGradientPadding];
        inputGradientSlice = new double[inputDepth][inputHeight][inputWidth];
    }

    public double[][][] forward(double[][][] input) {
        if (output == null)
            output = new double[numFilters][outputHeight][outputWidth];

        fillZeroes(output);
        forward(output, input);
        return output;
    }

    public void forward(double[][][] output, double[][][] input) {
        // our temporary array that is going to hold the convolution results for each filter
        if (output_slice == null)
            output_slice = new double[1][outputHeight][outputWidth];

        // reset the arrays before using them
        fillZeroes(stretched_input);
        fillZeroes(padded_and_stretched_input);
        fillZeroes(rotated_filters);

        this.input = input;

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            for (int fd = 0; fd < filterDepth; fd++) {
                rotate180(rotated_filters[filter_idx][fd], filters[filter_idx][fd]); // rotate filter and store in rotated_filters
            }
        }

        MiscUtils.addZeroesInBetween(stretched_input, input, 0, stride - 1, stride - 1);
        pad3d(padded_and_stretched_input, stretched_input, 0, 0, padding, padding + right_bottom_padding, padding, padding + right_bottom_padding);

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            fillZeroes(output_slice); // reset our output_slice array

            Convolution.convolve3d(output_slice, // our destination array
                    padded_and_stretched_input, rotated_filters[filter_idx], // convolution of the transformed_input and the rotated_filter
                    1, 1, 1); // our strides

            // copy the output_slice to the output array at filter_idx
            copyArray(output_slice[0], output[filter_idx]);
        }
    }

    public double[][][] backward(double[][][] outputGradient) {
        backward(inputGradient, outputGradient);
        return inputGradient;
    }

    public void backward(double[][][] inputGradient, double[][][] outputGradient) {
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


        // reset the inputGradient array
        fillZeroes(inputGradient);

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            fillZeroes(inputGradientSlice); // reset our inputGradientSlice array
            fillZeroes(paddedOutputGradientSlice); // reset our paddedOutputGradientSlice array

            pad3d(paddedOutputGradientSlice, // this is where the result of padding the array will be stored
                    new double[][][]{outputGradient[filter_idx]}, // this is what we want to pad
                    filterDepth - 1, filterDepth - 1,
                    outputGradientPadding, outputGradientPadding,
                    outputGradientPadding, outputGradientPadding);

            Convolution.convolve3d(
                    inputGradientSlice, // this is where the result of convolution will be stored
                    paddedOutputGradientSlice, // this is the input array which we are convolving the filters with
                    filters[filter_idx],
                    1, stride, stride);

            // now we have to store the inputGradientSlice in the inputGradient array
            // but we have to reverse it along the depth dimension.
            for (int c = 0; c < inputDepth; c++) {
                for (int i = 0; i < inputHeight; i++) {
                    for (int j = 0; j < inputWidth; j++) {
                        inputGradient[inputDepth - 1 - c][i][j] += inputGradientSlice[c][i][j];
                    }
                }
            }
        }
    }


    double[][][][] getFilterGradient(double[][][] outputGradient, double[][][] input) {
        /** calculates the filter gradients and bias gradients and updates the filters and biases

         apparently, filterGradients = convolve(input, outputGradient[k]) and biasGradients = sum(outputGradient[k])

         so that the output of convolution is of the same shape as the filterGradients, we have to pad the input accordingly
         */

        // reset the filtersGradient and stretched_input array
        fillZeroes(filtersGradient);
        fillZeroes(stretched_input);
//        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];
//        double[][][] stretched_input = MiscUtils.addZeroesInBetween(input, 0, stride - 1, stride - 1);

        MiscUtils.addZeroesInBetween(stretched_input, input, 0, stride - 1, stride - 1);

        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++) {
            double[][][] result = Convolution.convolve3d(
                    pad3d(new double[][][]{outputGradient[filter_idx]},
                            filterDepth - 1, filterDepth - 1,
                            outputGradientPadding, outputGradientPadding,
                            outputGradientPadding, outputGradientPadding
                    ),
                    stretched_input,
                    1, 1, 1);

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
            System.err.println("each filterGradient shape : " + filtersGradient[0].length + " " + filtersGradient[0][0].length + " " + filtersGradient[0][0][0].length);
            System.err.println("Supposed to be of shape : " + 1 + "x" + filterSize + "x" + filterSize);
            System.err.println("filterDepth : " + 1 + " outputDepth : " + outputDepth + " inputDepth : " + inputDepth);
        }

        return filtersGradient;
    }

    public void updateParametersBatch(double[][][][] outputGradients, double[][][][] inputs) {
        /** from a batch of inputs, for a batch of output gradients, we want to update based on the mean of the weights gradients and the mean of the biases gradients*/
        double[][][][][] filtersGradients = new double[outputGradients.length][numFilters][filterDepth][filterSize][filterSize];

        for (int sample_idx = 0; sample_idx < outputGradients.length; sample_idx++)
            copyArray(getFilterGradient(outputGradients[sample_idx], inputs[sample_idx]), filtersGradients[sample_idx]);

        filtersOptimizer.updateParameters(filters, MiscUtils.mean_1st_layer(filtersGradients));
    }

    public void updateParameters(double[][][] outputGradient, double[][][] input) {
        double[][][][] filtersGradient = getFilterGradient(outputGradient, input);

        filtersOptimizer.updateParameters(filters, filtersGradient);
    }


    @Deprecated
    public void updateParameters(double[][][] outputGradient) {
        updateParameters(outputGradient, this.input);
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

//        double[] biases = new double[1];
//        layer.biases = biases;

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

            layer.updateParameters(outputGradient);

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