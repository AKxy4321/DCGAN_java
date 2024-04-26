package DCGAN;

/*
 * Copyright (C) 2019 Elias Yilma
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//import DCGAN.Mat;

import java.util.Arrays;

public class Convolution {
    public int numFilters;
    public int filterSize;
    public int stride;
    public double[][][][] filters;
    public double[] biases;
    public double[][][] input3D;
    public int output_width, output_height, output_depth;
    public int input_width, input_height, input_depth;

    public int filter_depth;

    Convolution(int filterSize, int numFilters, int stride, int input_width, int input_height, int input_depth) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_depth = input_depth;

        this.output_width = (int) Math.floor((double) (input_width - filterSize) / stride + 1);
        this.output_height = (int) Math.floor((double) (input_height - filterSize) / stride + 1);
        this.output_depth = this.numFilters;

        this.filter_depth = input_depth;

//        this.filters = new double[numFilters][input_depth][filterSize][filterSize];
//        this.biases = new double[numFilters];

        this.filters = XavierInitializer.xavierInit4D(numFilters, filter_depth, filterSize);

        this.biases = XavierInitializer.xavierInit1D(numFilters);
    }


    public double[][][] forward(double[][][] input) {//TODO: recheck logic
        this.input3D = input;

        double[][][] output = new double[output_depth][output_height][output_width];

        for (int d = 0; d < this.output_depth; d++) {
            double[][][] f = this.filters[d];

            int y = 0; //-this.padding;
            for (int output_y = 0; output_y < this.output_height; y += stride, output_y++) {  // xy_stride
                int x = 0; // -this.padding;
                for (int output_x = 0; output_x < this.output_width; x += stride, output_x++) {  // xy_stride

                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    a += f[fd][fy][fx] * input[fd][input_y][input_x];
                                }
                            }
                        }
                    }
                    a += this.biases[d];
                    output[d][output_y][output_x] = a;
                }
            }
        }

        return output;
    }

    public double[][][] backprop(double[][][] output_gradients) {

        double[][][] inputGradient = new double[input_depth][input_height][input_width];


        for (int d = 0; d < this.output_depth; d++) {
            double[][][] f = this.filters[d];

            int y = 0; //-this.padding;
            for (int output_y = 0; output_y < this.output_height; y += stride, output_y++) {
                int x = 0; // -this.padding;
                for (int output_x = 0; output_x < this.output_width; x += stride, output_x++) {

                    // convolve centered at this particular location
                    double chain_grad = output_gradients[d][output_y][output_x];
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    inputGradient[fd][input_y][input_x] += f[fd][fy][fx] * chain_grad;

                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    public void updateParameters(double[][][] output_gradients, double learning_rate) {

        double[][][][] filterGradients = new double[numFilters][filter_depth][filterSize][filterSize];
        double[] biasGradients = new double[numFilters];

        for (int d = 0; d < this.output_depth; d++) {
            double[][][] f = this.filters[d];

            int y = 0; //-this.padding;
            for (int output_y = 0; output_y < this.output_height; y += stride, output_y++) {
                int x = 0; // -this.padding;
                for (int output_x = 0; output_x < this.output_width; x += stride, output_x++) {

                    // convolve centered at this particular location
                    double chain_grad = output_gradients[d][output_y][output_x];
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    filterGradients[d][fd][fy][fx] += input3D[fd][input_y][input_x] * chain_grad;
                                }
                            }
                        }
                    }
                    biasGradients[d] += chain_grad;
                }
            }
        }

        for (int k = 0; k < filters.length; k++) {
            for (int fd = 0; fd < filters[0].length; fd++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    for (int fx = 0; fx < filterSize; fx++) {
                        this.filters[k][fd][fy][fx] -= learning_rate * filterGradients[k][fd][fy][fx];
                    }
                }
            }
            this.biases[k] -= learning_rate * biasGradients[k];
        }

    }
}