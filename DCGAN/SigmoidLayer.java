package DCGAN;

public class SigmoidLayer{
    public double[][][] input;
    public double[][][] output;

    public double[][] input2D;
    public double[][] output2D;

    public double[] input1D;
    public double[] output1D;

    public double apply_sigmoid(double x) {
        return (double)(1/(1+Math.exp(-x)));
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
                    
                    output[d][h][w] = apply_sigmoid(input[d][h][w]);
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
                output2D[h][w] = apply_sigmoid(input[h][w]);
            }
        }
        return output2D;
    }

    public double[] forward(double[] input) {
        this.input1D = input;
        int size = input.length;
        output1D = new double[size];

        for (int i = 0; i < size; i++) {
            output1D[i] = apply_sigmoid(input[i]);
        }
        return output1D;
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

    public double[][] backward(double[][] d_L_d_out) {
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

    public double[] backward(double[] d_L_d_out) {
        double[] d_L_d_input = new double[d_L_d_out.length];
        int size = d_L_d_out.length;

        for (int i = 0; i < size; i++) {
            d_L_d_input[i] = (1-output1D[i]*output1D[i]) * d_L_d_out[i];
        }
        return d_L_d_input;
    }
}
/**
        for(int i=0; i<N; i++) {
            double v2wi = output.getWeight(i);
            input.setGradient(i, v2wi * (1.0 - v2wi) * output.getGradient(i));
        }
 */