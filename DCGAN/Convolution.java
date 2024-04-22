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
    public double[][][] filters;
    public double[][][] input3D;

    Convolution(int filterSize,int numFilters, int stride){
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.filters = init_filters();
    }

    Convolution(double[][][] filters, int stride){
        this.filters = filters;
        this.numFilters = this.filters.length;
        this.filterSize = this.filters[0].length;
        this.stride = stride;
    }

    public double[][] input;

    public double[][][] init_filters() {
        double[][][] result = new double[numFilters][filterSize][filterSize];
        for (int k = 0 ; k < numFilters ; k++){
            result[k] = Mat.m_zeros(filterSize, filterSize);
        }
        for (int k = 0; k < numFilters; k+=stride) {
            result[k] = Mat.m_random(filterSize, filterSize);
        }
        return result;
    }

    public double[][] convolveSxS(double[][] image, double[][] filter) {
        input = image;
        int resultSizeX = (image.length - filter.length) / stride + 1;
        int resultSizeY = (image[0].length - filter[0].length) / stride + 1;
        double[][] result = new double[resultSizeX][resultSizeY];
        for (int i = 0; i < resultSizeX; i++) {
            for (int j = 0; j < resultSizeY; j++) {
                int startX = i * stride;
                int startY = j * stride;
                double[][] conv_region = getConvolutionRegion(image, startX, startY, filter.length, filter[0].length);
                result[i][j] = Mat.mm_elsum(conv_region, filter);
            }
        }
        return result;
    }

    private double[][] getConvolutionRegion(double[][] image, int startX, int startY, int filterSizeX, int filterSizeY) {
        double[][] convRegion = new double[filterSizeX][filterSizeY];
        for (int i = 0; i < filterSizeX; i++) {
            for (int j = 0; j < filterSizeY; j++) {
                int x = startX + i;
                int y = startY + j;
                if (x >= 0 && x < image.length && y >= 0 && y < image[0].length) {
                    convRegion[i][j] = image[x][y];
                } else {
                    // Handle out-of-bounds access, for example, by padding with zeros
                    convRegion[i][j] = 0.0f; // Assuming double values
                }
            }
        }
        return convRegion;
    }


    public double[][][] forward(double[][] image) {
        int output_size = ((image.length - filterSize) / stride) + 1;
        double[][][] result = new double[numFilters][output_size][output_size];
        for (int k = 0; k < filters.length; k+=stride) {
            double[][] res = convolveSxS(image, filters[k]);
            result[k] = res;
        }
        return result;
    }

    public double[][][] forward(double[][][] image) {//TODO: recheck logic
        int output_size = ((image[0].length - filterSize) / stride) + 1;
        double[][][] result = new double[numFilters][output_size][output_size];
        for(int i = 0; i < image.length; i++) {
            for (int k = 0; k < filters.length; k += stride) {
                double[][] res = convolveSxS(image[i], filters[k]);
                result[k] = res;
            }
        }
        return result;
    }

    public double[][][] backprop(double[][][] d_L_d_out) {
        double[][][] d_L_d_filters = new double[filters.length][filters[0].length][filters[0][0].length];
        for (int i = 0; i <= (input.length - filterSize) / stride; i++) {
            for (int j = 0; j <= (input[0].length - filterSize) / stride; j++) {
                for (int k = 0; k < filters.length; k++) {
                    int startX = i * stride;
                    int startY = j * stride;
                    double[][] region = getConvolutionRegion(input, startX, startY, filterSize, filterSize);
                    d_L_d_filters[k] = Mat.mm_add(d_L_d_filters[k], Mat.m_scale(region, d_L_d_out[k][i][j]));
                }
            }
        }

        return d_L_d_filters;
    }

    public void updateParameters(double[][][] d_L_d_filters, double learning_rate) {
        for (int m = 0; m < filters.length; m++) {
            filters[m] = Mat.mm_add(filters[m], Mat.m_scale(d_L_d_filters[m], -learning_rate));
        }
    }

    public double[][][] get_Filters() {
        return filters;
    }
}