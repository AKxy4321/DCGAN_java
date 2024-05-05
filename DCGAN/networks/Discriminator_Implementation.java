package DCGAN.networks;

import DCGAN.UTIL;
import DCGAN.layers.Convolution;
import DCGAN.layers.DenseLayer;
import DCGAN.layers.LeakyReLULayer;
import DCGAN.layers.SigmoidLayer;

import java.util.Arrays;

public class Discriminator_Implementation {
    Convolution conv1;
//    BatchNormalization batch1;
    LeakyReLULayer leakyReLULayer1;
    Convolution conv2;
//    BatchNormalization batch2;
    LeakyReLULayer leakyReLULayer2;
    DenseLayer dense;
    SigmoidLayer sigmoidLayer;

    public Discriminator_Implementation() {
        int inputWidth = 28, inputHeight = 28;
        this.conv1 = new Convolution(5, 64, 3, inputWidth, inputHeight, 1);
        this.leakyReLULayer1 = new LeakyReLULayer();
        this.conv2 = new Convolution(5, 64, 3, conv1.output_width, conv1.output_height, conv1.output_depth);
        this.leakyReLULayer2 = new LeakyReLULayer();
        this.dense = new DenseLayer(conv2.output_depth * conv2.output_width * conv2.output_height, 1);
        this.sigmoidLayer = new SigmoidLayer();
    }

    public double[] getOutput(double[][] img) {
        double[][][] input = new double[1][][];
        input[0] = img;

        double[][][] output_conv1 = this.conv1.forward(input);
        double[][][] output_leakyRELU1 = this.leakyReLULayer1.forward(output_conv1);

        double[][][] output_conv2 = this.conv2.forward(output_leakyRELU1);
        double[][][] output_leakyRELU2 = this.leakyReLULayer2.forward(output_conv2);

        double[] output_leakyRELU_flattened = UTIL.flatten(output_leakyRELU2);
        double[] output_dense = this.dense.forward(output_leakyRELU_flattened);
        double[] discriminator_output = this.sigmoidLayer.forward(output_dense);
        return discriminator_output;
    }

    public double[][][] backward(double[] outputGradient) {
        double[] disc_in_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradient_sigmoid);

        double[][][] disc_in_gradient_dense_unflattened = UTIL.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
        double[][][] disc_in_gradient_leakyrelu2 = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened);
        double[][][] disc_in_gradient_conv2 = this.conv2.backprop(disc_in_gradient_leakyrelu2);

        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_conv2);
        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);

        // now we have the gradient of the loss function for the generated output wrt to the generator output(which is nothing but dou J / dou image)
        return disc_in_gradient_conv1; // this is the inputGradient
    }

    public void updateParameters(double[] outputGradient, double learning_rate_disc) {
        double[] disc_in_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradient_sigmoid);

        double[][][] disc_in_gradient_dense_unflattened = UTIL.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
        double[][][] disc_in_gradient_leakyrelu2 = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened);

        double[][][] disc_in_gradient_conv2 = this.conv2.backprop(disc_in_gradient_leakyrelu2);
        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_conv2);

        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);

        conv1.updateParameters(disc_in_gradient_leakyrelu1, learning_rate_disc);
        conv2.updateParameters(disc_in_gradient_leakyrelu2, learning_rate_disc);
        dense.updateParameters(disc_in_gradient_sigmoid, learning_rate_disc);

        // print the sum of all the gradients
        System.out.println("Sum of all gradients in discriminator: "
                + "\ndisc_in_gradient_dense : " + Arrays.stream(disc_in_gradient_dense).sum()
                + "\ndisc_in_gradient_leakyrelu2 : " + Arrays.stream(UTIL.flatten(disc_in_gradient_leakyrelu2)).sum()
                + "\ndisc_in_gradient_conv2 : " + Arrays.stream(UTIL.flatten(disc_in_gradient_conv2)).sum()
                + "\ndisc_in_gradient_leakyrelu1 : " + Arrays.stream(UTIL.flatten(disc_in_gradient_leakyrelu1)).sum()
                + "\ndisc_in_gradient_conv1 : " + Arrays.stream(UTIL.flatten(disc_in_gradient_conv1)).sum());
    }
}