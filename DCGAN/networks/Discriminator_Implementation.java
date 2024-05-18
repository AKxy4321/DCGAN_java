package DCGAN.networks;

import DCGAN.util.MiscUtils;
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

    public boolean verbose = false;
    int batchSize;

    public Discriminator_Implementation() {
        this(1, 1e-3);
    }

    public Discriminator_Implementation(int batchSize, double learning_rate) {
        this.batchSize = batchSize;
        int inputWidth = 28, inputHeight = 28;
        this.conv1 = new Convolution(4, 64, 2, inputWidth, inputHeight, 1, 0, 0, 0, learning_rate);
        this.leakyReLULayer1 = new LeakyReLULayer();
        this.conv2 = new Convolution(4, 128, 2, conv1.output_width, conv1.output_height, conv1.output_depth, 0, 0, 0, learning_rate);
        this.leakyReLULayer2 = new LeakyReLULayer();
        this.dense = new DenseLayer(conv2.output_depth * conv2.output_width * conv2.output_height, 1, learning_rate);
        this.sigmoidLayer = new SigmoidLayer();


        outputs_conv1 = new double[batchSize][][][];
        outputs_leakyRELU1 = new double[batchSize][][][];
        outputs_conv2 = new double[batchSize][][][];
        outputs_leakyRELU2 = new double[batchSize][][][];
        outputs_leakyRELU2_flattened = new double[batchSize][];
        outputs_dense = new double[batchSize][];
        outputs_sigmoid = new double[batchSize][];
    }

    public double[] getOutput(double[][] black_and_white_image) {
        double[][][] input = new double[1][][];
        input[0] = black_and_white_image;

        return getOutput(input);
    }


    public double[] getOutput(double[][][] input) {
        double[][][] output_conv1 = this.conv1.forward(input);
        double[][][] output_leakyRELU1 = this.leakyReLULayer1.forward(output_conv1);

        double[][][] output_conv2 = this.conv2.forward(output_leakyRELU1);
        double[][][] output_leakyRELU2 = this.leakyReLULayer2.forward(output_conv2);

        double[] output_leakyRELU2_flattened = MiscUtils.flatten(output_leakyRELU2);
        double[] output_dense = this.dense.forward(output_leakyRELU2_flattened);
        double[] output_sigmoid = this.sigmoidLayer.forward(output_dense);
        return output_sigmoid;
    }

    public double[][][] backward(double[] outputGradient) {
        double[] disc_in_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradient_sigmoid);

        double[][][] disc_in_gradient_dense_unflattened = MiscUtils.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
        double[][][] disc_in_gradient_leakyrelu2 = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened);

        double[][][] disc_in_gradient_conv2 = this.conv2.backprop(disc_in_gradient_leakyrelu2);
        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_conv2);

        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);

        // now we have the gradient of the loss function for the generated output wrt to the generator output(which is nothing but dou J / dou image)
        return disc_in_gradient_conv1; // this is the inputGradient
    }

    public void updateParameters(double[] outputGradient) {
        double[] disc_in_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradient_sigmoid);

        double[][][] disc_in_gradient_dense_unflattened = MiscUtils.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
        double[][][] disc_in_gradient_leakyrelu2 = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened);

        double[][][] disc_in_gradient_conv2 = this.conv2.backprop(disc_in_gradient_leakyrelu2);
        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_conv2);

        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);

        conv1.updateParameters(disc_in_gradient_leakyrelu1);
        conv2.updateParameters(disc_in_gradient_leakyrelu2);
        dense.updateParameters(disc_in_gradient_sigmoid);


        if (verbose) {
            // print the sum of all the gradients
            System.out.println("Sum of all gradients in discriminator: "
                    + "\ndisc_in_gradient_dense : " + Arrays.stream(disc_in_gradient_dense).sum()
                    + "\ndisc_in_gradient_leakyrelu2 : " + Arrays.stream(MiscUtils.flatten(disc_in_gradient_leakyrelu2)).sum()
                    + "\ndisc_in_gradient_conv2 : " + Arrays.stream(MiscUtils.flatten(disc_in_gradient_conv2)).sum()
                    + "\ndisc_in_gradient_leakyrelu1 : " + Arrays.stream(MiscUtils.flatten(disc_in_gradient_leakyrelu1)).sum()
                    + "\ndisc_in_gradient_conv1 : " + Arrays.stream(MiscUtils.flatten(disc_in_gradient_conv1)).sum());
        }
    }

    double[][][][] inputs;
    double[][][][] outputs_conv1;
    double[][][][] outputs_leakyRELU1;
    double[][][][] outputs_conv2;
    double[][][][] outputs_leakyRELU2;
    double[][] outputs_leakyRELU2_flattened;
    double[][] outputs_dense;
    double[][] outputs_sigmoid;

    public double[][] forwardBatch(double[][][][] inputs) {
        this.inputs = inputs;

        for (int i = 0; i < inputs.length; i++) {
            outputs_conv1[i] = this.conv1.forward(inputs[i]);
            outputs_leakyRELU1[i] = this.leakyReLULayer1.forward(outputs_conv1[i]);

            outputs_conv2[i] = this.conv2.forward(outputs_leakyRELU1[i]);
            outputs_leakyRELU2[i] = this.leakyReLULayer2.forward(outputs_conv2[i]);

            outputs_leakyRELU2_flattened[i] = MiscUtils.flatten(outputs_leakyRELU2[i]);
            outputs_dense[i] = this.dense.forward(outputs_leakyRELU2_flattened[i]);
            outputs_sigmoid[i] = this.sigmoidLayer.forward(outputs_dense[i]);
        }

        return outputs_sigmoid;
    }

    public double[][][][] backwardBatch(double[][] outputGradients) {
        double[][][][] disc_in_gradients_conv1 = new double[batchSize][][][];

        for (int i = 0; i < batchSize; i++) {
            double[] disc_in_gradients_sigmoid = this.sigmoidLayer.backward(outputGradients[i], outputs_sigmoid[i]);
            double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradients_sigmoid);

            double[][][] disc_in_gradient_dense_unflattened = MiscUtils.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
            double[][][] disc_in_gradients_leakyrelu2 = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened, outputs_leakyRELU2[i]);

            double[][][] disc_in_gradient_conv2 = this.conv2.backprop(disc_in_gradients_leakyrelu2);
            double[][][] disc_in_gradients_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_conv2, outputs_leakyRELU1[i]);

            disc_in_gradients_conv1[i] = this.conv1.backprop(disc_in_gradients_leakyrelu1);
        }

        return disc_in_gradients_conv1;
    }

    public void updateParametersBatch(double[][] outputGradients) {
        double[][] disc_in_gradients_sigmoid = new double[batchSize][];
        double[][][][] disc_in_gradients_leakyrelu2 = new double[batchSize][][][];
        double[][][][] disc_in_gradients_leakyrelu1 = new double[batchSize][][][];

        for (int i = 0; i < batchSize; i++) {
            disc_in_gradients_sigmoid[i] = this.sigmoidLayer.backward(outputGradients[i], outputs_sigmoid[i]);
            double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradients_sigmoid[i]);

            double[][][] disc_in_gradient_dense_unflattened = MiscUtils.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
            disc_in_gradients_leakyrelu2[i] = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened, outputs_leakyRELU2[i]);

            double[][][] disc_in_gradient_conv2 = this.conv2.backprop(disc_in_gradients_leakyrelu2[i]);
            disc_in_gradients_leakyrelu1[i] = this.leakyReLULayer1.backward(disc_in_gradient_conv2, outputs_leakyRELU1[i]);

//            not needed
//            double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradients_leakyrelu1[i]);
        }

        dense.updateParametersBatch(disc_in_gradients_sigmoid, outputs_leakyRELU2_flattened);
        conv2.updateParametersBatch(disc_in_gradients_leakyrelu2, outputs_leakyRELU1);
        conv1.updateParametersBatch(disc_in_gradients_leakyrelu1, inputs);

    }
}