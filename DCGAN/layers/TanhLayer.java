package DCGAN.layers;

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
        this.output = new double[depth][height][width];

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
        this.output2D = new double[height][width];

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output2D[h][w] = Math.tanh(input[h][w]);
            }
        }
        return output2D;
    }


    @Deprecated
    public double[][][] backward(double[][][] outputGradient) {
        return backward(output, outputGradient);
    }

    public double[][][] backward(double[][][] outputGradient, double[][][] output) {
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
//                    System.out.println(output[d][h][w]);
                    inputGradient[d][h][w] = (1 - output[d][h][w] * output[d][h][w]) * outputGradient[d][h][w];
                }
            }
        }
        return inputGradient;
    }

    public double[][] backprop(double[][] outputGradient) {
        double[][] inputGradient = new double[outputGradient.length][outputGradient[0].length];
        int height = outputGradient.length;
        int width = outputGradient[0].length;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                inputGradient[h][w] = (1 - output2D[h][w] * output2D[h][w]) * outputGradient[h][w];
            }
        }
        return inputGradient;
    }

    public static void main(String[] args) {
        double[][][] input = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
        TanhLayer tanh = new TanhLayer();
        double[][][] output = tanh.forward(input);
        double[][][] outputGradient = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
        double[][][] inputGradient = tanh.backward(outputGradient);
        System.out.println("Output: ");
        for (int d = 0; d < output.length; d++) {
            for (int h = 0; h < output[0].length; h++) {
                for (int w = 0; w < output[0][0].length; w++) {
                    System.out.print(output[d][h][w] + " ");
                }
                System.out.println();
            }
        }
        System.out.println("Input Gradient: ");
        for (int d = 0; d < inputGradient.length; d++) {
            for (int h = 0; h < inputGradient[0].length; h++) {
                for (int w = 0; w < inputGradient[0][0].length; w++) {
                    System.out.print(inputGradient[d][h][w] + " ");
                }
                System.out.println();
            }
        }
    }

}