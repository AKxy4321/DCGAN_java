package DCGAN;

public class LeakyReLULayer {

    public double[][][] input;
    public double[][][] output;

    public double[][] input2D;
    public double[][] output2D;

    public double[] input1D;
    public double[] output1D;

    double k=0.001f;

    public LeakyReLULayer() {
        this.k = 0.001f; // default value
    }

    public LeakyReLULayer(double k) {
        this.k = k;
    }

    public double apply_leaky_relu(double x) {
        return x > 0 ? x : x * k;
    }

    public double[][][] forward(double[][][] input) {
        this.input = input;
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        output = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[d][h][w] = apply_leaky_relu(input[d][h][w]);
                }
            }
        }

        return output;
    }

    public double[] forward(double[] input) {
        this.input1D = input;
        int depth = input.length;
        output1D = new double[depth];

        for (int d = 0; d < depth; d++) {
            output1D[d] = apply_leaky_relu(input[d]);
        }

        return output1D;
    }

    public double[][] forward(double[][] input) {
        this.input2D = input;
        int height = input.length;
        int width = input[0].length;
        output2D = new double[height][width];

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output2D[h][w] = apply_leaky_relu(input[h][w]);
            }
        }
        return output2D;
    }

    public double[][][] backward(double[][][] d_L_d_out) {
        double[][][] d_L_d_input = new double[d_L_d_out.length][d_L_d_out[0].length][d_L_d_out[0][0].length];
        int depth = d_L_d_out.length;
        int height = d_L_d_out[0].length;
        int width = d_L_d_out[0][0].length;

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // If the input value was negative, the gradient is k;
                    d_L_d_input[d][h][w] = output[d][h][w] > 0 ? d_L_d_out[d][h][w] : k * d_L_d_out[d][h][w];
                }
            }
        }

        return d_L_d_input;
    }

    public double[][] backward(double[][] d_L_d_out) {
        double[][] d_L_d_input = new double[d_L_d_out.length][d_L_d_out[0].length];
        int height = d_L_d_out.length;
        int width = d_L_d_out[0].length;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // If the input value was negative, the gradient is k;
                // TODO: Check if this is correct
                d_L_d_input[h][w] = output2D[h][w] > 0 ? d_L_d_out[h][w] : k * d_L_d_out[h][w];
            }
        }
        /**
         * correct algorithm is :
         * for(int i=0; i<len(inputs); i++) {
         * * if(output.getWeight(i) <= 0) {
         * * * * input.setGradient(i, k*output.getGradient(i))); // threshold
         * * } else {
         * * * * input.setGradient(i, output.getGradient(i));
         * * }
         * }
         */
        return d_L_d_input;
    }

    public double[] backward(double[] d_L_d_out) {
        double[] d_L_d_input = new double[d_L_d_out.length];
        int height = d_L_d_out.length;

        for (int h = 0; h < height; h++) {
            d_L_d_input[h] = output1D[h] > 0 ? d_L_d_out[h] : k * d_L_d_out[h];
        }

        return d_L_d_input;
    }
}