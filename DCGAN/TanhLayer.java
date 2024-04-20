package DCGAN;

public class TanhLayer {

    public float[][][] input;
    public float[][][] output;

    public float[][] input2D;
    public float[][] output2D;


    public float[][][] forward(float[][][] input) {
        this.input = input;
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        output = new float[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[d][h][w] = (float)Math.tanh(input[d][h][w]);
                }
            }
        }

        return output;
    }

    public float[][] forward(float[][] input) {
        this.input2D = input;
        int height = input.length;
        int width = input[0].length;
        output2D = new float[height][width];

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output2D[h][w] = (float)Math.tanh(input[h][w]);
            }
        }
        return output2D;
    }

    public float[][][] backward(float[][][] d_L_d_out) {
        float[][][] d_L_d_input = new float[d_L_d_out.length][d_L_d_out[0].length][d_L_d_out[0][0].length];
        int depth = d_L_d_out.length;
        int height = d_L_d_out[0].length;
        int width = d_L_d_out[0][0].length;

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    d_L_d_input[d][h][w] =  (1-output[d][h][w]*output[d][h][w]) * d_L_d_out[d][h][w];
                }
            }
        }
        return d_L_d_input;
    }

    public float[][] backprop(float[][] d_L_d_out) {
        float[][] d_L_d_input = new float[d_L_d_out.length][d_L_d_out[0].length];
        int height = d_L_d_out.length;
        int width = d_L_d_out[0].length;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                d_L_d_input[h][w] = (1-output2D[h][w]*output2D[h][w]) *d_L_d_out[h][w];
            }
        }
        return d_L_d_input;
    }

}
/**
        for(int i=0; i<input.weights.length; i++) {
            double v2wi = output.getWeight(i);
            input.setGradient(i,(1.0 - v2wi * v2wi) * output.getGradient(i));
        }
*/