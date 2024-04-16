package cnn;

import UTIL.Mat;

public class Sigmoid {
    public float[][] weights;
    public float[][] input;
    public float[][] bias;
    public float[][] output;
    int filters;
    int side;

    public Sigmoid(int input, int output, int filters, int side) {
        this.weights = Mat.m_scale(Mat.m_random(input, output), 1.0f / input);
        this.bias = Mat.v_zeros(10);
        this.filters = filters;
        this.side = side;
    }

    public Sigmoid(float[][] weights, float[][] bias){
        this.bias = bias;
        this.weights = weights;
    }

    public float[][] forward(float[][][] input) {
        float[][] in = Mat.m_flatten(input);
        output = new float[1][bias.length];
        output = Mat.mm_add(Mat.mm_mult(in, weights), bias);
        // Apply sigmoid activation function instead of softmax
        output = sigmoid(output);
        this.input = in;
        return output;
    }

    public float[][][] backprop(float[][] d_L_d_out, float learning_rate) {
        float[][] d_L_d_t = new float[1][d_L_d_out[0].length];
        float[][] sigmoid_derivative = sigmoidDerivative(output);
        float[][] d_L_d_inputs = null;

        for (int i = 0; i < d_L_d_out[0].length; i++) {
            float grad = d_L_d_out[0][i];
            if (grad == 0) {
                continue;
            }
            float sigmoid_grad = sigmoid_derivative[0][i];
            d_L_d_t[0][i] = grad * sigmoid_grad;

            float[][] d_t_d_weight = Mat.m_transpose(input);
            float[][] d_t_d_inputs = weights;
            float[][] d_L_d_w = Mat.mm_mult(d_t_d_weight, d_L_d_t);
            d_L_d_inputs = Mat.mm_mult(d_t_d_inputs, Mat.m_transpose(d_L_d_t));
            weights = Mat.mm_add(Mat.m_scale(d_L_d_w, -learning_rate), weights);
            bias = Mat.mm_add(Mat.m_scale(d_L_d_t, -learning_rate), bias);
        }
        return Mat.reshape(Mat.m_transpose(d_L_d_inputs), filters, side, side);
    }

    // Sigmoid activation function
    private float[][] sigmoid(float[][] x) {
        float[][] result = new float[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = (float) (1 / (1 + Math.exp(-x[i][j])));
            }
        }
        return result;
    }

    // Derivative of sigmoid activation function
    private float[][] sigmoidDerivative(float[][] x) {
        float[][] sigmoid = sigmoid(x);
        float[][] result = new float[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = sigmoid[i][j] * (1 - sigmoid[i][j]);
            }
        }
        return result;
    }
}
