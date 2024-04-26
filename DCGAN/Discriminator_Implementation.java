package DCGAN;

import java.util.Arrays;

public class Discriminator_Implementation {
    Convolution conv1;
    LeakyReLULayer leakyReLULayer1;
    MaxPool maxPool;
    DenseLayer dense;
    SigmoidLayer sigmoidLayer;

    public Discriminator_Implementation() {
        int inputWidth = 28, inputHeight = 28;
        this.conv1 = new Convolution(5, 64, 3, inputWidth, inputHeight, 1);
        this.leakyReLULayer1 = new LeakyReLULayer();
        this.maxPool = new MaxPool();
        this.dense = new DenseLayer(conv1.output_depth * conv1.output_width * conv1.output_height / 4, 1);
        this.sigmoidLayer = new SigmoidLayer();
    }

    public double[] getOutput(double[][] img) {
        double[][][] input = new double[1][][];
        input[0] = img;
        double[][][] output_conv1 = this.conv1.forward(input);
        double[][][] output_leakyRELU1 = this.leakyReLULayer1.forward(output_conv1);
        double[][][] output_maxpool = this.maxPool.forward(output_leakyRELU1);
        double[] output_maxpool_flattened = UTIL.flatten(output_maxpool);
        double[] output_dense = this.dense.forward(output_maxpool_flattened);
        double[] discriminator_output = this.sigmoidLayer.forward(output_dense);
        return discriminator_output;
    }

    public double[][][] backward(double[] outputGradient) {
        double[] disc_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_gradient_dense = this.dense.backward(disc_gradient_sigmoid);

        double[][][] disc_in_gradient_dense_unflattened = UTIL.unflatten(disc_gradient_dense, leakyReLULayer1.output.length, leakyReLULayer1.output[0].length/2, leakyReLULayer1.output[0][0].length/2);

        double[][][] disc_in_gradient_maxpool = this.maxPool.backprop(disc_in_gradient_dense_unflattened);
        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_maxpool);
        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);
        // now we have the gradient of the loss function for the generated output wrt to the generator output(which is nothing but dou J / dou image)
        return disc_in_gradient_conv1; // this is the inputGradient
    }

    public void updateParameters(double[] outputGradient, double learning_rate_disc) {
        double[] disc_in_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradient_sigmoid);

        this.dense.updateParameters(disc_in_gradient_sigmoid, learning_rate_disc);

        double[][][] disc_in_gradient_dense_unflattened = UTIL.unflatten(disc_in_gradient_dense, leakyReLULayer1.output.length, leakyReLULayer1.output[0].length/2, leakyReLULayer1.output[0][0].length/2);

        double[][][] disc_in_gradient_maxpool = this.maxPool.backprop(disc_in_gradient_dense_unflattened);
        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_maxpool);
        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);

        conv1.updateParameters(disc_in_gradient_leakyrelu1, learning_rate_disc);

//        // print the first filter weights
//        double[][][][] filters = conv1.filters;
//        for(int i=0;i<filters[0].length;i++){
//            for(int j=0;j<filters[0][0].length;j++){
//                System.out.println(Arrays.toString(filters[0][i][j]));
//            }
//        }

        // print the sum of all the gradients
        System.out.println("Sum of all gradients in discriminator: "
                + Arrays.stream(UTIL.flatten(disc_in_gradient_conv1)).sum()
                + "\n" + Arrays.stream(UTIL.flatten(disc_in_gradient_leakyrelu1)).sum()
                + "\n" + Arrays.stream(disc_in_gradient_dense).sum()
                + "\n" + Arrays.stream(disc_in_gradient_sigmoid).sum() + "\n");
    }
}