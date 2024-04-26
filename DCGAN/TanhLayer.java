package DCGAN;

public class TanhLayer {

    public double[][][] input;
    public double[][][] output;

    public double[][] input2D;
    public double[][] output2D;

    public double[][][] forward(double[][][] input) {
        this.input = input;
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        output = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[d][h][w] = Math.tanh(input[d][h][w]);
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
                output2D[h][w] = Math.tanh(input[h][w]);
            }
        }
        return output2D;
    }

    public double[][][] backward(double[][][] outputGradient) {
        double[][][] inputGradient = new double[outputGradient.length][outputGradient[0].length][outputGradient[0][0].length];
        int depth = outputGradient.length;
        int height = outputGradient[0].length;
        int width = outputGradient[0][0].length;
//        System.out.println("For TanhLayer:");
//        System.out.println("inputGradient shape : "+inputGradient.length+" "+inputGradient[0].length+" "+inputGradient[0][0].length);
//        System.out.println("outputGradient shape : "+outputGradient.length+" "+outputGradient[0].length+" "+outputGradient[0][0].length);
//        System.out.println("output shape : "+output.length+" "+output[0].length+" "+output[0][0].length);

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    inputGradient[d][h][w] =  (1-output[d][h][w]*output[d][h][w]) * outputGradient[d][h][w];
                }
            }
        }
        return inputGradient;
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