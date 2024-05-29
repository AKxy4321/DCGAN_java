package DCGAN.networks;

import DCGAN.layers.Convolution;
import DCGAN.layers.DenseLayer;
import DCGAN.layers.LeakyReLULayer;
import DCGAN.optimizers.OptimizerHyperparameters;
import DCGAN.util.MiscUtils;

import java.io.Serializable;

import static DCGAN.util.MathUtils.mean;
import static DCGAN.util.MiscUtils.clamp;

public class Critic implements Serializable {
    private static final long serialVersionUID = 1L;
    Convolution conv1;
    //    BatchNormalization batch1;
    LeakyReLULayer leakyReLULayer1;
    Convolution conv2;
    //    BatchNormalization batch2;
    LeakyReLULayer leakyReLULayer2;
    DenseLayer dense;

    public boolean verbose = false;
    int batchSize;
    private double min_clip;
    private double max_clip;


    public Critic(int batchSize, OptimizerHyperparameters optimizerHyperparameters) {
        this.batchSize = batchSize;
        int inputWidth = 28, inputHeight = 28;
        this.conv1 = new Convolution(4, 64, 2,
                inputWidth, inputHeight, 1,
                3, 3, 0, optimizerHyperparameters);
        this.leakyReLULayer1 = new LeakyReLULayer(0.2);
        this.conv2 = new Convolution(4, 128, 2,
                conv1.outputWidth, conv1.outputHeight, conv1.outputDepth,
                3, 3, 0, optimizerHyperparameters);
        this.leakyReLULayer2 = new LeakyReLULayer(0.2);
        this.dense = new DenseLayer(conv2.outputDepth * conv2.outputWidth * conv2.outputHeight, 1, optimizerHyperparameters);


        outputs_conv1 = new double[batchSize][][][];
        outputs_leakyRELU1 = new double[batchSize][][][];
        outputs_conv2 = new double[batchSize][][][];
        outputs_leakyRELU2 = new double[batchSize][][][];
        outputs_leakyRELU2_flattened = new double[batchSize][];
        outputs_dense = new double[batchSize][];
    }

    public void setClip(double min_clip, double max_clip) {
        this.min_clip = min_clip;
        this.max_clip = max_clip;
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

        double[] output_dense = this.dense.forward(MiscUtils.flatten(output_leakyRELU2));
        return output_dense;
    }

    double[][][][] inputs;
    double[][][][] outputs_conv1;
    double[][][][] outputs_leakyRELU1;
    double[][][][] outputs_conv2;
    double[][][][] outputs_leakyRELU2;
    double[][] outputs_leakyRELU2_flattened;
    double[][] outputs_dense;

    public double[][] forwardBatch(double[][][][] inputs) {
        this.inputs = inputs;

        for (int i = 0; i < inputs.length; i++, System.out.print(verbose ? " " + i : "")) {
            outputs_conv1[i] = this.conv1.forward(inputs[i]);
            outputs_leakyRELU1[i] = this.leakyReLULayer1.forward(outputs_conv1[i]);

            outputs_conv2[i] = this.conv2.forward(outputs_leakyRELU1[i]);
            outputs_leakyRELU2[i] = this.leakyReLULayer2.forward(outputs_conv2[i]);

            outputs_leakyRELU2_flattened[i] = MiscUtils.flatten(outputs_leakyRELU2[i]);
            outputs_dense[i] = this.dense.forward(outputs_leakyRELU2_flattened[i]);
        }

        return outputs_dense;
    }

    public double[][][][] backwardBatch(double[][] outputGradients) {
        double[][][][] disc_in_gradients_conv1 = new double[outputGradients.length][][][];

        for (int i = 0; i < outputGradients.length; i++, System.out.print(verbose ? " " + i : "")) {
            double[] disc_in_gradient_dense = this.dense.backward(outputGradients[i]);

            double[][][] disc_in_gradient_dense_unflattened = MiscUtils.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
            double[][][] disc_in_gradients_leakyrelu2 = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened, outputs_leakyRELU2[i]);

            double[][][] disc_in_gradient_conv2 = this.conv2.backward(disc_in_gradients_leakyrelu2);
            double[][][] disc_in_gradients_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_conv2, outputs_leakyRELU1[i]);

            disc_in_gradients_conv1[i] = this.conv1.backward(disc_in_gradients_leakyrelu1);
        }

        return disc_in_gradients_conv1;
    }

    public void updateParametersBatch(double[][] outputGradients) {
        double[][][][] disc_in_gradients_leakyrelu2_conv2_out_grad = new double[batchSize][][][];
        double[][][][] disc_in_gradients_leakyrelu1_conv1_out_grad = new double[batchSize][][][];

        for (int i = 0; i < batchSize; i++) {
            double[] disc_in_gradient_dense = this.dense.backward(outputGradients[i]);

            double[][][] disc_in_gradient_dense_unflattened = MiscUtils.unflatten(disc_in_gradient_dense, leakyReLULayer2.output.length, leakyReLULayer2.output[0].length, leakyReLULayer2.output[0][0].length);
            disc_in_gradients_leakyrelu2_conv2_out_grad[i] = this.leakyReLULayer2.backward(disc_in_gradient_dense_unflattened, outputs_leakyRELU2[i]);

            double[][][] disc_in_gradient_conv2 = this.conv2.backward(disc_in_gradients_leakyrelu2_conv2_out_grad[i]);
            disc_in_gradients_leakyrelu1_conv1_out_grad[i] = this.leakyReLULayer1.backward(disc_in_gradient_conv2, outputs_leakyRELU1[i]);

//            not needed
//            double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradients_leakyrelu1[i]);
        }

        dense.updateParametersBatch(outputGradients, outputs_leakyRELU2_flattened);
        conv2.updateParametersBatch(disc_in_gradients_leakyrelu2_conv2_out_grad, outputs_leakyRELU1);
        conv1.updateParametersBatch(disc_in_gradients_leakyrelu1_conv1_out_grad, inputs);

        // clip the weights
        MiscUtils.clipInPlace(dense.weights, min_clip, max_clip); // Apparently clipping should be done for only convolution layers
        MiscUtils.clipInPlace(conv2.filters, min_clip, max_clip);
        MiscUtils.clipInPlace(conv1.filters, min_clip, max_clip);

        if(verbose){
            // mean of gradients of each layer
            System.out.println("Mean of output gradients of each layer of critic");
            System.out.println("dense layer: " + mean(MiscUtils.flatten(outputGradients)));
            System.out.println("conv2: " + mean(MiscUtils.flatten(disc_in_gradients_leakyrelu2_conv2_out_grad[0])));
            System.out.println("conv1: " + mean(MiscUtils.flatten(disc_in_gradients_leakyrelu1_conv1_out_grad[0])));
        }
    }
}