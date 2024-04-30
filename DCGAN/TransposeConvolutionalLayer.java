package DCGAN;

import java.util.Arrays;

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

            int y = 0;
            for (int inputY = this.padding; inputY < inputHeight - this.padding; y += stride, inputY++) {  // xy_stride
                int x = 0;
                for (int inputX = this.padding; inputX < inputWidth - this.padding; x += stride, inputX++) {  // xy_stride

                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int k = 0; k < numFilters; k++) {

                        for (int fy = 0; fy < filterSize; fy++) {
                            int outputY = y + fy * stride; // coordinates in the original input array coordinates
                            for (int fx = 0; fx < filterSize; fx++) {

                                int outputX = x + fx * stride;
                                if (outputY >= 0 && outputY < outputHeight && outputX >= 0 && outputX < outputWidth) {
                                    for (int fd = 0; fd < filters[0].length; fd++) {// filter depth
                                        double[][][] f = this.filters[k];

                                        //calculate the 180 degree rotated filter indices
                                        int new_fx = fx;//filterSize - 1 - fx;
                                        int new_fy = fy;//filterSize - 1 - fy;

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

    public void updateParameters(double[][][] outputGradient, double[][][] input, double learningRate) {
        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];

        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {

                        for (int h = 0; h < outputHeight; h++) {
                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;

                                //calculate the 180 degree rotated outputGradient indices
                                int new_W = w;//outputWidth - 1 - w;
                                int new_H = h;//outputHeight - 1 - h;

                                if ((this.padding <= inH && inH < inputHeight - this.padding)
                                        && (this.padding <= inW && inW < inputWidth - this.padding)) {
                                    filtersGradient[k][c][i][j] += outputGradient[k][new_H][new_W] * input[c][inH][inW];
                                }
                            }
                        }

                    }
                }
            }
        }

        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        this.filters[k][c][i][j] -= learningRate * filtersGradient[k][c][i][j];
                    }
                }
            }
        }
    }

    public void updateParameters(double[][][] outputGradient, double learningRate) {
        double[][][][] filtersGradient = new double[numFilters][filterDepth][filterSize][filterSize];

        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {

                        for (int h = 0; h < outputHeight; h++) {
                            for (int w = 0; w < outputWidth; w++) {
                                int inH = h - i * this.stride;
                                int inW = w - j * this.stride;

                                //calculate the 180 degree rotated outputGradient indices
                                int new_W = w;//outputWidth - 1 - w;
                                int new_H = h;//outputHeight - 1 - h;

                                if ((this.padding <= inH && inH < inputHeight - this.padding)
                                        && (this.padding <= inW && inW < inputWidth - this.padding)) {
                                    filtersGradient[k][c][i][j] += outputGradient[k][new_H][new_W] * input[c][inH][inW];
                                }
                            }
                        }

                    }
                }
            }
        }

        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < filterDepth; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        this.filters[k][c][i][j] -= learningRate * filtersGradient[k][c][i][j];
                    }
                }
            }
        }
    }


    public static void main(String[] args) {

        DenseLayer dense = new DenseLayer(5, 2 * 2);
        SigmoidLayer lu = new SigmoidLayer();
        DenseLayer dense2 = new DenseLayer(2 * 2, 2 * 2);
//        output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding
        TransposeConvolutionalLayer tconv = new TransposeConvolutionalLayer(2, 1, 2, 2, 2, 1, 0);


        double[][] targetOutput = {
                {0, 0, 0, 1},
                {0, 3, 2, 3},
                {0, 2, 0, 3},
                {4, 6, 6, 9}
        };

        double[] input = XavierInitializer.xavierInit1D(5);

        for (int epoch = 0; epoch < 5000; epoch++) {

            //forward pass
            double[] dense_output = dense.forward(input);
            double[] leaky_output = lu.forward(dense_output);
            double[] dense2_output = dense2.forward(leaky_output);
            double[][] tconv_output = tconv.forward(UTIL.unflatten(dense2_output, tconv.inputDepth, tconv.inputHeight, tconv.inputWidth))[0];


            double[][] outputGradient = new double[tconv_output.length][tconv_output[0].length];
            UTIL.calculateGradientMSE(outputGradient, tconv_output, targetOutput);
            double mse = UTIL.lossMSE(tconv_output, targetOutput);

            System.out.println("Epoch " + (epoch + 1) + ", MSE: " + mse);

            // backward pass
            double[][][] tconv_in_gradient = tconv.backward(new double[][][]{outputGradient});
            double[] dense2_in_gradient = dense2.backward(UTIL.flatten(tconv_in_gradient));
            double[] lu_in_gradient = lu.backward(dense2_in_gradient);
            double[] dense_gradient = dense.backward(lu_in_gradient);

            dense.updateParameters(lu_in_gradient, 0.05);
            dense2.updateParameters(dense2_in_gradient, 0.05);
            tconv.updateParameters(new double[][][]{outputGradient}, 0.05);


//            System.out.println("Sum of values in each gradient :");
//            System.out.println("Dense gradient: " + Arrays.stream(dense_gradient).sum());
//            System.out.println("Leaky gradient: " + Arrays.stream(lu_in_gradient).sum());
//            System.out.println("Dense2 gradient: " + Arrays.stream(dense2_in_gradient).sum());
//            System.out.println("Tconv gradient: " + Arrays.stream(UTIL.flatten(tconv_in_gradient)).sum());


            if (epoch == 4999) {
                System.out.println("Output:");
                for (int i = 0; i < tconv_output.length; i++) {
                    for (int j = 0; j < tconv_output[0].length; j++) {
                        System.out.print(tconv_output[i][j] + " ");
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