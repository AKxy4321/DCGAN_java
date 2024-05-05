package DCGAN.layers;

public class LeakyReLULayer {

    public double[][][] input;
    public double[][][] output;

    public double[][] input2D;
    public double[][] output2D;

    public double[] input1D;
    public double[] output1D;

    double k = 0.01;

    public LeakyReLULayer(double k){
        this.k = k;
    }
    public LeakyReLULayer(){
        this.k = 0.01;
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

    public double[][][] backward(double[][][] outputGradient) {
        double[][][] inputGradient = new double[outputGradient.length][outputGradient[0].length][outputGradient[0][0].length];
        int depth = outputGradient.length;
        int height = outputGradient[0].length;
        int width = outputGradient[0][0].length;

//        System.out.println("For LeakyRELULayer:");
//        System.out.println("inputGradient shape : "+inputGradient.length+" "+inputGradient[0].length+" "+inputGradient[0][0].length);
//        System.out.println("outputGradient shape : "+outputGradient.length+" "+outputGradient[0].length+" "+outputGradient[0][0].length);
//        System.out.println("output shape : "+output.length+" "+output[0].length+" "+output[0][0].length);

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    inputGradient[d][h][w] = output[d][h][w] > 0 ? outputGradient[d][h][w] : k * outputGradient[d][h][w];
                }
            }
        }

        return inputGradient;
    }

    public double[][] backward(double[][] outputGradient) {
        double[][] d_L_d_input = new double[outputGradient.length][outputGradient[0].length];
        int height = outputGradient.length;
        int width = outputGradient[0].length;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                d_L_d_input[h][w] = output2D[h][w] > 0 ? outputGradient[h][w] : k * outputGradient[h][w];
            }
        }
        return d_L_d_input;
    }

    public double[] backward(double[] outputGradient) {
        double[] d_L_d_input = new double[outputGradient.length];
        int height = outputGradient.length;

        for (int h = 0; h < height; h++) {
            d_L_d_input[h] = output1D[h] > 0 ? outputGradient[h] : k * outputGradient[h];
        }

        return d_L_d_input;
    }
}
