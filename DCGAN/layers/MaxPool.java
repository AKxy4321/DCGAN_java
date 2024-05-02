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
package DCGAN.layers;

import DCGAN.Mat;

public class MaxPool {
        public double[][][] input;
        public double[][][] output;

    public double[][] max_pool(double[][] img) {
        double[][] pool = new double[img.length / 2][img[0].length / 2];
        for (int i = 0; i < pool.length - 1; i++) {
            for (int j = 0; j < pool[0].length - 1; j++) {
                pool[i][j] = Mat.m_max(Mat.m_sub(img, i * 2, i * 2 + 1, j * 2, j * 2 + 1));
            }
        }
        return pool;
    }

    public double[][][] forward(double[][][] dta) {
        input = dta;
        int outputSizeX = dta[0].length / 2;
        int outputSizeY = dta[0][0].length / 2;
        double[][][] result = new double[dta.length][outputSizeX][outputSizeY];
        for (int k = 0; k < dta.length; k++) {
            double[][] res = max_pool(dta[k]);
            result[k] = res;
        }
        output = result;
        return result;
    }

    public double[][][] backprop(double[][][] d_L_d_out) {
        double[][][] d_L_d_input = new double[input.length][input[0].length][input[0][0].length];
        int outputSizeX = input[0].length / 2;
        int outputSizeY = input[0][0].length / 2;
        for (int i = 0; i < outputSizeX; i++) {
            for (int j = 0; j < outputSizeY; j++) {
                for (int k = 0; k < d_L_d_out.length; k++) {
                    int startX = i * 2;
                    int startY = j * 2;
                    double[][] region = Mat.m_sub(input[k], startX, startX + 1, startY, startY + 1);
                    for (int m = 0; m < region.length; m++) {
                        for (int n = 0; n < region[0].length; n++) {
                            if (Math.abs(output[k][i][j] - region[m][n]) < 0.00000001) {
                                d_L_d_input[k][startX + m][startY + n] = d_L_d_out[k][i][j];
                            }
                        }
                    }
                }
            }
        }
        return d_L_d_input;
    }

}
