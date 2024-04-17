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
package cnn;

import UTIL.Mat;

public class Dense {
    public float[][] weights;
    public float[][] input;
    public float[][] bias;
    public float[][] output;
    int filters;
    int side;

    public Dense(int input, int output, int filters, int side) {
        this.weights = Mat.m_scale(Mat.m_random(input, output), 1.0f / input);
        this.bias = Mat.v_zeros(10);
        this.filters = filters;
        this.side = side;
    }

    public Dense(float[][] weights, float[][] bias){
        this.bias = bias;
        this.weights = weights;
    }

    public float[][] forward(float[][][] input) {
        float[][] in = Mat.m_flatten(input);
        output = new float[1][bias.length];
        output = Mat.mm_add(Mat.mm_mult(in, weights), bias);
        this.input = in;
        return output; // Return the raw output without applying any activation function
    }

    public float[][][] backprop(float[][][] d_L_d_out, float learning_rate) {
        int batchSize = d_L_d_out.length;
        int inputSize = input[0].length;
        int outputSize = output[0].length;

        float[][][] d_L_d_inputs = new float[batchSize][input.length][input[0].length];
        float[][] d_L_d_t = new float[batchSize][outputSize];

        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < inputSize; i++) {
                float sumOutput = Mat.v_sum(output);
                for (int j = 0; j < outputSize; j++) {
                    float grad = d_L_d_out[b][i][j];
                    if (grad == 0) {
                        continue;
                    }
                    float d_out_d_t = output[b][i] * (sumOutput - output[b][i]) / (sumOutput * sumOutput);
                    d_L_d_t[b][j] += grad * d_out_d_t;
                    for (int k = 0; k < input.length; k++) {
                        for (int l = 0; l < input[0].length; l++) {
                            d_L_d_inputs[b][k][l] += weights[i][j] * grad * d_out_d_t;
                        }
                    }
                    // Update weights and bias here
                    weights[i][j] -= learning_rate * grad * d_out_d_t * input[b][i];
                    bias[0][j] -= learning_rate * grad * d_out_d_t;
                }
            }
        }
        return d_L_d_inputs;
    }
}
