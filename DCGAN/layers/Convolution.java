package DCGAN.layers;

import DCGAN.optimizers.AdamOptimizer;
import DCGAN.util.MiscUtils;
import DCGAN.util.XavierInitializer;

import java.util.logging.Logger;

public class Convolution {
    public int numFilters;
    public int filterSize;
    public int stride;
    public double[][][][] filters;
    public double[] biases;
    public double[][][] input3D;
    public int outputWidth, outputHeight, outputDepth;
    public int input_width, input_height, input_depth;

    public int filter_depth;
    public int inputPaddingX, inputPaddingY;
    public int padding = 0;

    AdamOptimizer filtersOptimizer, biasesOptimizer;
    public boolean use_bias = false;

    private static Logger logger = Logger.getLogger(Convolution.class.getName());

    public Convolution(int filterSize, int numFilters, int stride, int input_width, int input_height, int input_depth) {
        this(filterSize, numFilters, stride, input_width, input_height, input_depth, 0, 0, 0);
    }

    public Convolution(int filterSize, int numFilters, int stride, int input_width, int input_height, int input_depth, int inputPaddingX, int inputPaddingY, int inputPaddingZ) {
        this(filterSize, numFilters, stride, input_width, input_height, input_depth, inputPaddingX, inputPaddingY, inputPaddingZ, 0.001);
    }

    public Convolution(int filterSize, int numFilters, int stride, int input_width, int input_height, int input_depth, int inputPaddingX, int inputPaddingY, int inputPaddingZ, double learning_rate) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_depth = input_depth;

        // output_shape = ((input_height - kernel_size + 2 * padding) / stride) + 1 for symmetrical padding
        this.padding = inputPaddingX;
        this.outputWidth = (int) Math.floor((double) (input_width - filterSize + 2 * padding) / stride + 1);
        this.outputHeight = (int) Math.floor((double) (input_height - filterSize + 2 * padding) / stride + 1);
        this.outputDepth = this.numFilters;

        // since filterDepth isn't mentioned, its convention to assume that the filter depth is equal to the input depth,
        // although it makes sense to make its depth 1, that goes against convention,
        // but usually we want 3d output, although if you put filter_depth to something smaller, you will get a 4d output which we don't really wanna work with
        this.filter_depth = input_depth;

        this.filters = new double[numFilters][filter_depth][][];
        for (int filter_idx = 0; filter_idx < numFilters; filter_idx++)
            filters[filter_idx] = XavierInitializer.xavierInit3D(filter_depth, filterSize, filterSize);
        this.biases = XavierInitializer.xavierInit1D(numFilters);

        this.inputPaddingX = inputPaddingX;
        this.inputPaddingY = inputPaddingY;

        filtersOptimizer = new AdamOptimizer(numFilters * filter_depth * filterSize * filterSize, learning_rate, 0.5, 0.999, 1e-8);
        biasesOptimizer = new AdamOptimizer(numFilters, learning_rate, 0.5, 0.999, 1e-8);
    }


    public double[][][] forward(double[][][] input) {
        this.input3D = input;

        double[][][] output = new double[outputDepth][outputHeight][outputWidth];

        // preprocessing : pad the input
        input = pad3d(input, 0, padding, padding);

        for (int d = 0; d < this.outputDepth; d++) {
            // we know the convolution result will have a depth of 1 because we are convolving a 3d input with a filter of the same depth
            // od = ((id - fd) / stride_in_z) + 1 = ((id - id) / stride_in_z) + 1 = 0 + 1 = 1

            // we take this 1xoutput_heightxoutput_width and reshape it to 2d : output_heightxoutput_width
            output[d] = convolve3d(input, filters[d], 1, stride, stride)[0];


            if (use_bias) {
                for (int y = 0; y < outputHeight; y++)
                    for (int x = 0; x < outputWidth; x++)
                        output[d][y][x] += biases[d];
            }
        }

        return output;
    }

    public double[][][] backward(double[][][] output_gradients) {

        /**
         returns the input Gradient. Input gradient is calculated using the formula:
         inputGradient = transpose_convolution(output_gradients, filters)
         *
         * */

        double[][][] inputGradient = new double[input_depth][input_height][input_width];

        for (int d = 0; d < this.outputDepth; d++) {
            double[][][] f = this.filters[d];

            int y = -padding;
            for (int output_y = 0; output_y < this.outputHeight; y += stride, output_y++) {
                int x = -padding;
                for (int output_x = 0; output_x < this.outputWidth; x += stride, output_x++) {

                    // convolve centered at this particular location
                    double chain_grad = output_gradients[d][output_y][output_x];
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    inputGradient[fd][input_y][input_x] += f[fd][filterSize - 1 - fy][filterSize - 1 - fx] * chain_grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    public double[][][][] calculateFiltersGradient(double[][][] output_gradient, double[][][] input3D) {
        double[][][][] filtersGradient = new double[numFilters][filter_depth][filterSize][filterSize];

        for (int filter_idx = 0; filter_idx < this.outputDepth; filter_idx++) {
            double[][][] f = this.filters[filter_idx];

            int y = -padding;
            for (int output_y = 0; output_y < this.outputHeight; y += stride, output_y++) {
                int x = -padding;
                for (int output_x = 0; output_x < this.outputWidth; x += stride, output_x++) {

                    // convolve centered at this particular location
                    double chain_grad = output_gradient[filter_idx][output_y][output_x];
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    filtersGradient[filter_idx][fd][fy][fx] += input3D[fd][input_y][input_x] * chain_grad;
                                }
                            }
                        }
                    }
                }
            }
        }
        return filtersGradient;
    }

    public double[] calculateBiasGradient(double[][][] output_gradient) {
        double[] biasGradient = new double[numFilters];
        for (int filter_idx = 0; filter_idx < this.outputDepth; filter_idx++) {
            double[][][] f = this.filters[filter_idx];

            int y = 0;
            for (int output_y = 0; output_y < this.outputHeight; y += stride, output_y++) {
                int x = 0;
                for (int output_x = 0; output_x < this.outputWidth; x += stride, output_x++) {
                    biasGradient[filter_idx] += output_gradient[filter_idx][output_y][output_x];
                }
            }
        }
        return biasGradient;
    }

    @Deprecated
    public void updateParameters(double[][][] output_gradient) {
        updateParameters(output_gradient, this.input3D);
    }

    public void updateParameters(double[][][] output_gradient, double[][][] input3D) {
        /**
         * updates the filter weights and the biases
         * the filterGradients are calculated using the formula : df = convolution(input, output_gradients)
         *
         * biasGradient[d] = sum(output_gradients[d])
         * */

        double[][][][] filtersGradient = calculateFiltersGradient(output_gradient, input3D);
        double[] biasGradient = calculateBiasGradient(output_gradient);

        filtersOptimizer.updateParameters(filters, filtersGradient);
        biasesOptimizer.updateParameters(biases, biasGradient);
    }

    public void updateParametersBatch(double[][][][] output_gradients, double[][][][] inputs) {
        double[][][][][] filtersGradients = new double[output_gradients.length][][][][];
        double[][] biasGradients = new double[output_gradients.length][];

        for (int i = 0; i < output_gradients.length; i++) {
            filtersGradients[i] = calculateFiltersGradient(output_gradients[i], inputs[i]);
            biasGradients[i] = calculateBiasGradient(output_gradients[i]);
        }

        filtersOptimizer.updateParameters(filters, MiscUtils.mean_1st_layer(filtersGradients));
        biasesOptimizer.updateParameters(biases, MiscUtils.mean_1st_layer(biasGradients));
    }


    public static double[][] pad2d(double[][] input, int heightPadding, int widthPadding) {
        int input_width = input[0].length;
        int input_height = input.length;

        int paddedInputHeight = input_height + heightPadding * 2;
        int paddedInputWidth = input_width + widthPadding * 2;

        // Pad the input with zeros
        double[][] paddedInput = new double[paddedInputHeight][paddedInputWidth];
        for (int h = 0; h < input_height; h++) {
            for (int w = 0; w < input_width; w++) {
                paddedInput[h + heightPadding][w + widthPadding] = input[h][w];
            }
        }

        return paddedInput;
    }

    public static double[][][] pad3d(double[][][] input, int depthPadding, int heightPadding, int widthPadding) {
        int input_width = input[0][0].length;
        int input_height = input[0].length;
        int input_depth = input.length;

        int paddedInputWidth = input_width + widthPadding * 2;
        int paddedInputHeight = input_height + heightPadding * 2;
        int paddedInputDepth = input_depth + depthPadding * 2;

        double[][][] paddedInput = new double[paddedInputDepth][paddedInputHeight][paddedInputWidth];

        for (int d = 0; d < input_depth; d++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < input_width; w++) {
                    paddedInput[d + depthPadding][h + heightPadding][w + widthPadding] = input[d][h][w];
                }
            }
        }

        return paddedInput;
    }

    public static double[][][] pad3d(double[][][] paddedAndStretchedInput, int front, int back, int up, int bottom, int left, int right) {
        int input_depth = paddedAndStretchedInput.length;
        int input_height = paddedAndStretchedInput[0].length;
        int input_width = paddedAndStretchedInput[0][0].length;

        int paddedInputDepth = input_depth + front + back;
        int paddedInputHeight = input_height + up + bottom;
        int paddedInputWidth = input_width + left + right;

        double[][][] paddedInput = new double[paddedInputDepth][paddedInputHeight][paddedInputWidth];

        for (int d = 0; d < input_depth; d++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < input_width; w++) {
                    paddedInput[d + front][h + up][w + left] = paddedAndStretchedInput[d][h][w];
                }
            }
        }

        return paddedInput;
    }


    public static double[][][] pad3d(double[][][] input, int padding) {
        return pad3d(input, padding, padding, padding);
    }

    public static double[][] pad2d(double[][] input, int padding) {
        return pad2d(input, padding, padding);
    }

    public static double[][][] convolve3d(double[][][] input, double[][][] filter, int stride) {
        return convolve3d(input, filter, stride, stride, stride);
    }


    public static double[][][] convolve3d(double[][][] input, double[][][] filter, int z_stride, int y_stride, int x_stride) {
//        System.out.println("Input before padding:");
//        UTIL.prettyprint(input);

        int input_width = input[0][0].length;
        int input_height = input[0].length;
        int input_depth = input.length;

//        System.out.println("Padded input : ");
//        UTIL.prettyprint(input);


        int filterHeight = filter[0].length;
        int filterWidth = filter[0][0].length;
        int filterDepth = filter.length;

        // output_shape = ((input_height - kernel_size + 2 * padding) / stride) + 1
        // padding is already applied, so we can omit it in the below formula
        int output_width = (int) Math.floor((double) (input_width - filterWidth) / x_stride + 1);
        int output_height = (int) Math.floor((double) (input_height - filterHeight) / y_stride + 1);
        int output_depth = (int) Math.floor((double) (input_depth - filterDepth) / z_stride + 1);

        double[][][] output = new double[output_depth][output_height][output_width];


        int z = 0;
        for (int output_z = 0; output_z < output_depth; z += z_stride, output_z++) {
            int y = 0;
            for (int output_y = 0; output_y < output_height; y += y_stride, output_y++) {  // xy_stride
                int x = 0;
                for (int output_x = 0; output_x < output_width; x += x_stride, output_x++) {  // xy_stride

                    // Here we are taking dot product with the filter and the overlapping input region
                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int fz = 0; fz < filterDepth; fz++) {
                        int input_z = z + fz;
                        for (int fy = 0; fy < filterHeight; fy++) {
                            int input_y = y + fy; // coordinates in the original input array coordinates
                            for (int fx = 0; fx < filterWidth; fx++) {
                                int input_x = x + fx;
                                if (input_y >= 0 && input_y < input_height &&
                                        input_x >= 0 && input_x < input_width &&
                                        input_z >= 0 && input_z < input_depth) {
                                    a += filter[fz][fy][fx] * input[input_z][input_y][input_x];
                                }
                            }
                        }
                    }

                    output[output_z][output_y][output_x] = a;

                }
            }
        }

        return output;
    }

    public static double[][] convolve2d(double[][] input, double[][] filter, int stride) {
        return convolve2d(input, filter, stride, 0, 0);
    }

    public static double[][] convolve2d(double[][] input, double[][] filter, int stride, int inputPadding) {
        return convolve2d(input, filter, stride, inputPadding, inputPadding);
    }

    public static double[][] convolve2d(double[][] input, double[][] filter, int stride, int heightPadding, int widthPadding) {
//        System.out.println("Input before padding:");
//        UTIL.prettyprint(input);

        if (widthPadding > 0 || heightPadding > 0)
            input = pad2d(input, heightPadding, widthPadding);

        int input_width = input[0].length;
        int input_height = input.length;

//        System.out.println("Padded input : ");
//        UTIL.prettyprint(input);

        int filterSize = filter[0].length;

        // output_shape = ((input_height - kernel_size + 2 * padding) / stride) + 1
        // padding is already applied, so we can omit it in the below formula
        int output_width = (int) Math.floor((double) (input_width - filterSize) / stride + 1);
        int output_height = (int) Math.floor((double) (input_height - filterSize) / stride + 1);
        double[][] output = new double[output_height][output_width];


        int y = 0;
        for (int output_y = 0; output_y < output_height; y += stride, output_y++) {  // xy_stride
            int x = 0;
            for (int output_x = 0; output_x < output_width; x += stride, output_x++) {  // xy_stride

                // convolve centered at this particular location
                double a = 0.0;
                for (int fy = 0; fy < filterSize; fy++) {
                    int input_y = y + fy; // coordinates in the original input array coordinates
                    for (int fx = 0; fx < filterSize; fx++) {
                        int input_x = x + fx;
                        if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                            a += filter[fy][fx] * input[input_y][input_x];
                        }
                    }
                }
                output[output_y][output_x] = a;
            }
        }

        return output;
    }

    public static void main(String[] args) {
        MiscUtils.prettyprint(
                convolve2d(new double[][]{
                                {1, 1},
                                {1, 1}},
                        new double[][]{
                                {1, 1},
                                {1, 1}},
                        1, 1));
    }
}