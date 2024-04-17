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

public class SoftMax {
        public float[][] weights;
        public float[][] input;
        public float[][] bias;
        public float[][] output;
        int filters;
        int side;

    public SoftMax(int input, int output, int filters, int side) {
        this.weights = Mat.m_scale(Mat.m_random(input, output), 1.0f / input);
        this.bias = Mat.v_zeros(10);
        this.filters = filters;
        this.side = side;
    }

    public SoftMax(float[][] weights, float[][] bias){
        this.bias = bias;
        this.weights = weights;
    }

    public float[][] forward(float[][][] input) {
        float[][] in = Mat.m_flatten(input);
        output = new float[1][bias.length];
        output = Mat.mm_add(Mat.mm_mult(in, weights), bias);
        float[][] totals = Mat.v_exp(output);
        float inv_activation_sum = 1 / Mat.v_sum(totals);
        this.input = in;
        return Mat.v_scale(totals, inv_activation_sum);
    }

    public float[][][] backprop(float[][] d_L_d_out, float learning_rate) {
        float[][] d_L_d_t = new float[1][d_L_d_out[0].length];
        float[][] t_exp = Mat.v_exp(output);
        float S = Mat.v_sum(t_exp);
        float[][] d_L_d_inputs=null;
        
        for (int i = 0; i < d_L_d_out[0].length; i++) {
            float grad = d_L_d_out[0][i];
            if (grad == 0) {
                continue;
            }
            float[][] d_out_d_t = Mat.v_scale(t_exp, -t_exp[0][i] / (S * S));
            d_out_d_t[0][i] = t_exp[0][i] * (S - t_exp[0][i]) / (S * S);
            
            d_L_d_t = Mat.m_scale(d_out_d_t, grad);
            float[][] d_t_d_weight = Mat.m_transpose(input);
            float[][] d_t_d_inputs = weights;
            float[][] d_L_d_w = Mat.mm_mult(d_t_d_weight, d_L_d_t);
            d_L_d_inputs = Mat.mm_mult(d_t_d_inputs, Mat.m_transpose(d_L_d_t));
            float[][] d_L_d_b = d_L_d_t;
            weights = Mat.mm_add(Mat.m_scale(d_L_d_w, -learning_rate), weights);
            bias = Mat.mm_add(Mat.m_scale(d_L_d_b, -learning_rate), bias);
        }
        return Mat.reshape(Mat.m_transpose(d_L_d_inputs), filters, side, side);
    }
}
