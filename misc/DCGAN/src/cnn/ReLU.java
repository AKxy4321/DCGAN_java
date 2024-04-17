package cnn;

public class ReLU {
    public float[][][] input;
    public float[][][] output;

    public float[][][] forward(float[][][] input) {
        this.input = input;
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        output = new float[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[d][h][w] = Math.max(0, input[d][h][w]); // ReLU activation function
                }
            }
        }
        return output;
    }

    public float[][][] backprop(float[][][] d_L_d_out) {
        float[][][] d_L_d_input = new float[d_L_d_out.length][d_L_d_out[0].length][d_L_d_out[0][0].length];
        int depth = d_L_d_out.length;
        int height = d_L_d_out[0].length;
        int width = d_L_d_out[0][0].length;

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // If the input value was negative, the gradient is 0; otherwise, it passes through
                    d_L_d_input[d][h][w] = input[d][h][w] > 0 ? d_L_d_out[d][h][w] : 0;
                }
            }
        }
        return d_L_d_input;
    }
}
