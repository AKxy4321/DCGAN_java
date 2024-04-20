package DCGAN;

public class TanhLayer {

    public double[][][] input;
    public double[][][] output;

    public double[][] input2D;
    public double[][] output2D;

    //custom tanh function
    public double tanh(double x) {
        return (Math.pow(Math.E, x) - Math.pow(Math.E, -x)) / (Math.pow(Math.E, x) + Math.pow(Math.E, -x));
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
                    output[d][h][w] = (double)Math.tanh(input[d][h][w]);
                }
            }
        }

        return output;
    }

    public double[][] forward(double[][] input) {
        this.input2D = input;
        int height = input.length;
        int width = input[0].length;
        output2D = new double[height][width];

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output2D[h][w] = (double)Math.tanh(input[h][w]);
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
                    d_L_d_input[d][h][w] =  (1-output[d][h][w]*output[d][h][w]) * d_L_d_out[d][h][w];
                }
            }
        }
        return d_L_d_input;
    }

    public double[][] backprop(double[][] d_L_d_out) {
        double[][] d_L_d_input = new double[d_L_d_out.length][d_L_d_out[0].length];
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