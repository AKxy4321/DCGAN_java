package DCGAN.networks;

import DCGAN.optimizers.AdamHyperparameters;
import DCGAN.optimizers.OptimizerHyperparameters;
import DCGAN.util.MiscUtils;
import DCGAN.util.ArrayInitializer;
import DCGAN.layers.*;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import static DCGAN.util.MathUtils.mean;
import static DCGAN.util.MiscUtils.*;
import static DCGAN.util.SerializationUtils.saveObject;
import static DCGAN.util.TrainingUtils.calculateGradientRMSE;
import static DCGAN.util.TrainingUtils.lossRMSE;

public class Generator_Implementation_Without_Batchnorm implements Serializable {
    int dense_output_size;
    DenseLayer dense;
    LeakyReLULayer leakyReLU1;
    TransposeConvolutionalLayer tconv1;
    LeakyReLULayer leakyReLU2;
    TransposeConvolutionalLayer tconv2;
    LeakyReLULayer leakyReLU3;
    TransposeConvolutionalLayer tconv3;
    TanhLayer tanh;

    public boolean verbose = false;
    public int batchSize;
    int noise_length = 100;
    Random random = new Random(1);

    OptimizerHyperparameters optimizerHyperparameters;

    public Generator_Implementation_Without_Batchnorm(int batchSize, OptimizerHyperparameters optimizerHyperparameters) {
        this.optimizerHyperparameters = optimizerHyperparameters;

        this.batchSize = batchSize;

        int tconv1_input_width = 4, tconv1_input_height = 4, tconv1_input_depth = 256;
        this.dense_output_size = tconv1_input_width * tconv1_input_height * tconv1_input_depth;
        this.dense = new DenseLayer(noise_length, this.dense_output_size, optimizerHyperparameters);
        this.leakyReLU1 = new LeakyReLULayer(0.1);

        // this.stride * (inputHeight - 1) + filterSize - 2 * padding;
        this.tconv1 = new TransposeConvolutionalLayer(3, 128, 2,
                tconv1_input_width, tconv1_input_height, tconv1_input_depth,
                1, 0, 0, 1, false, optimizerHyperparameters);
        assert tconv1.outputHeight == 7;
        assert tconv1.outputWidth == 7;
        this.leakyReLU2 = new LeakyReLULayer(0.1);

        this.tconv2 = new TransposeConvolutionalLayer(4, 64, 2,
                tconv1.outputWidth, tconv1.outputHeight, tconv1.outputDepth,
                2, 0, 0, 1, false, optimizerHyperparameters);
        assert tconv2.outputHeight == 14;
        assert tconv2.outputWidth == 14;
        this.leakyReLU3 = new LeakyReLULayer(0.1);

        this.tconv3 = new TransposeConvolutionalLayer(4, 1, 2,
                tconv2.outputWidth, tconv2.outputHeight, tconv2.outputDepth,
                2, 0, 0, 1, false, optimizerHyperparameters);
        assert tconv3.outputHeight == 28;
        assert tconv3.outputWidth == 28;
        // (i + 4 - 1) + 3 - 2*0 = i + 6
        this.tanh = new TanhLayer();


        noises = new double[batchSize][dense.inputSize];
        denseOutputs = new double[batchSize][dense.outputSize];
        denseOutputsUnflattened = new double[batchSize][tconv1.inputDepth][tconv1.inputHeight][tconv1.inputWidth];
        leakyReLU1Outputs = new double[batchSize][tconv1.inputDepth][tconv1.inputHeight][tconv1.inputWidth];

        tconv1Outputs = new double[batchSize][tconv1.outputDepth][tconv1.outputHeight][tconv1.outputWidth];
        leakyReLU2Outputs = new double[batchSize][tconv1.outputDepth][tconv1.outputHeight][tconv1.outputWidth];

        tconv2Outputs = new double[batchSize][tconv2.outputDepth][tconv2.outputHeight][tconv2.outputWidth];
        leakyReLU3Outputs = new double[batchSize][tconv2.outputDepth][tconv2.outputHeight][tconv2.outputWidth];

        tconv3Outputs = new double[batchSize][tconv3.outputDepth][tconv3.outputHeight][tconv3.outputWidth];
        tanhOutputs = new double[batchSize][tconv3.outputDepth][tconv3.outputHeight][tconv3.outputWidth];
    }


    public OptimizerHyperparameters getOptimizerHyperparameters() {
        return optimizerHyperparameters;
    }

    public void setOptimizerHyperparameters(OptimizerHyperparameters optimizerHyperparameters) {
        this.optimizerHyperparameters = optimizerHyperparameters;
        this.dense.setOptimizerHyperparameters(optimizerHyperparameters);
        this.tconv1.setOptimizerHyperparameters(optimizerHyperparameters);
        this.tconv2.setOptimizerHyperparameters(optimizerHyperparameters);
        this.tconv3.setOptimizerHyperparameters(optimizerHyperparameters);
    }

    public double[][][] generateImage() {
        // using a spherical Z
        double[] noise = ArrayInitializer.initGaussian1D(noise_length);

        double[] gen_dense_output = this.dense.forward(noise);
        double[][][] gen_dense_output_unflattened = MiscUtils.unflatten(gen_dense_output, tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);
        double[][][] gen_leakyrelu_output1 = this.leakyReLU1.forward(gen_dense_output_unflattened);
//        saveImage(getBufferedImage(gen_leakyrelu_output1), "gen_leakyrelu_output1.png");
        double[][][] outputTconv1 = this.tconv1.forward(gen_leakyrelu_output1);
        double[][][] gen_leakyrelu_output2 = this.leakyReLU2.forward(outputTconv1);
//        saveImage(getBufferedImage(gen_leakyrelu_output2), "gen_leakyrelu_output2.png");
        double[][][] outputTconv2 = this.tconv2.forward(gen_leakyrelu_output2);
        double[][][] gen_leakyrelu_output3 = this.leakyReLU3.forward(outputTconv2);
//        saveImage(getBufferedImage(gen_leakyrelu_output3), "gen_leakyrelu_output3.png");
        double[][][] gen_tconv3_output = this.tconv3.forward(gen_leakyrelu_output3);
        double[][][] fakeImage = this.tanh.forward(gen_tconv3_output);
        return fakeImage;
    }

    double[][] noises;
    double[][] denseOutputs;
    double[][][][] denseOutputsUnflattened;
    double[][][][] leakyReLU1Outputs;
    double[][][][] tconv1Outputs;
    double[][][][] leakyReLU2Outputs;
    double[][][][] tconv2Outputs;
    double[][][][] leakyReLU3Outputs;
    double[][][][] tconv3Outputs;
    double[][][][] tanhOutputs;

    public double[][][][] forwardBatch() {
        for (int i = 0; i < batchSize; i++, System.out.print(verbose ? " " + i : "")) {
            ArrayInitializer.initGaussian1D(noises[i]);
            denseOutputs[i] = dense.forward(noises[i]);

            denseOutputsUnflattened[i] = MiscUtils.unflatten(denseOutputs[i], tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);

            leakyReLU1Outputs[i] = leakyReLU1.forward(denseOutputsUnflattened[i]);

            tconv1.forward(tconv1Outputs[i], leakyReLU1Outputs[i]); // do forward pass and store the result in tconv1Outputs[i]

            leakyReLU2Outputs[i] = leakyReLU2.forward(tconv1Outputs[i]);

            tconv2.forward(tconv2Outputs[i], leakyReLU2Outputs[i]); // do forward pass and store the result in tconv2Outputs[i]

            leakyReLU3Outputs[i] = leakyReLU3.forward(tconv2Outputs[i]);

            tconv3.forward(tconv3Outputs[i], leakyReLU3Outputs[i]); // do forward pass and store the result in tconv3Outputs[i]

            tanhOutputs[i] = tanh.forward(tconv3Outputs[i]);
        }

        if (verbose) {
            MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv1Outputs[0][0])), "tconv1Outputs[0][0].png");
            MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv2Outputs[0][0])), "tconv2Outputs[0][0].png");
            MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv3Outputs[0][0])), "tconv3Outputs[0][0].png");
        }


        return tanhOutputs;
    }

    public void updateParametersBatch(double[][][][] outputGradients) {
        // in_gradient means the gradient of the loss function with respect to the input of this layer
        // outgrad means the gradient of the loss function with respect to the output of this layer

        double[][][][] tconv1_in_gradient_l1_outgrad = new double[batchSize][tconv1.inputDepth][tconv1.inputHeight][tconv1.inputWidth];
        double[][][][] tconv2_in_gradient_l2_outgrad = new double[batchSize][tconv2.inputDepth][tconv2.inputHeight][tconv2.inputWidth];
        double[][][][] tconv3_in_gradient_l3_outgrad = new double[batchSize][tconv3.inputDepth][tconv3.inputHeight][tconv3.inputWidth];

        double[][] leakyrelu_in_gradient_d_outgrad_flattened = new double[batchSize][];
        double[][][][] leakyrelu2_in_gradient_t1_outgrad = new double[batchSize][][][];
        double[][][][] leakyrelu3_in_gradient_t2_outgrad = new double[batchSize][][][];

        double[][][][] tanh_in_gradient_t3_outgrad = new double[batchSize][][][];


        for (int i = 0; i < batchSize; i++, System.out.print(verbose ? " " + i : "")) {
            tanh_in_gradient_t3_outgrad[i] = this.tanh.backward(outputGradients[i], tanhOutputs[i]);
            this.tconv3.backward(tconv3_in_gradient_l3_outgrad[i], tanh_in_gradient_t3_outgrad[i]); // do backward and store the input gradient in the first array
            leakyrelu3_in_gradient_t2_outgrad[i] = this.leakyReLU3.backward(tconv3_in_gradient_l3_outgrad[i], leakyReLU3Outputs[i]);

            this.tconv2.backward(tconv2_in_gradient_l2_outgrad[i], leakyrelu3_in_gradient_t2_outgrad[i]); // do backward and store the input gradient in the first array
            leakyrelu2_in_gradient_t1_outgrad[i] = this.leakyReLU2.backward(tconv2_in_gradient_l2_outgrad[i], leakyReLU2Outputs[i]);

            this.tconv1.backward(tconv1_in_gradient_l1_outgrad[i], leakyrelu2_in_gradient_t1_outgrad[i]); // do backward and store the input gradient in the first array
            double[][][] leakyrelu_in_gradient_d_outgrad = this.leakyReLU1.backward(tconv1_in_gradient_l1_outgrad[i], leakyReLU1Outputs[i]);

            leakyrelu_in_gradient_d_outgrad_flattened[i] = MiscUtils.flatten(leakyrelu_in_gradient_d_outgrad);
        }

        this.tconv1.updateParametersBatch(leakyrelu2_in_gradient_t1_outgrad, leakyReLU1Outputs);
        this.tconv2.updateParametersBatch(leakyrelu3_in_gradient_t2_outgrad, leakyReLU2Outputs);
        this.tconv3.updateParametersBatch(tanh_in_gradient_t3_outgrad, leakyReLU3Outputs);
        this.dense.updateParametersBatch(leakyrelu_in_gradient_d_outgrad_flattened, noises);

        if (verbose) {
            // print out the sum of each gradient by flattening it and summing it up using stream().sum()
            System.out.println("Sum of each gradient in generator: ");
            for (int i = 0; i < batchSize; i++) {
                System.out.println("tanh_in_gradient: " + Arrays.stream(MiscUtils.flatten(tanh_in_gradient_t3_outgrad[i])).sum());
                System.out.println("leakyrelu3_in_gradient: " + Arrays.stream(MiscUtils.flatten(leakyrelu3_in_gradient_t2_outgrad[i])).sum());
                System.out.println("tconv2_in_gradient: " + Arrays.stream(MiscUtils.flatten(tconv2_in_gradient_l2_outgrad[i])).sum());
                System.out.println("leakyrelu2_in_gradient: " + Arrays.stream(MiscUtils.flatten(leakyrelu2_in_gradient_t1_outgrad[i])).sum());
                System.out.println("tconv1_in_gradient: " + Arrays.stream(MiscUtils.flatten(tconv1_in_gradient_l1_outgrad[i])).sum());
                System.out.println("leakyrelu_in_gradient: " + Arrays.stream(MiscUtils.flatten(leakyrelu_in_gradient_d_outgrad_flattened)).sum());
                System.out.println("leakyrelu_in_gradient_flattened: " + Arrays.stream(leakyrelu_in_gradient_d_outgrad_flattened[i]).sum());

                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tanh_in_gradient_t3_outgrad[0][0])), "outputs/tanh_in_gradient_t3_outgrad[0][0].png");
                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv3_in_gradient_l3_outgrad[0][0])), "outputs/tconv3_in_gradient_l3_outgrad[0][0].png");
                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv2_in_gradient_l2_outgrad[0][0])), "outputs/tconv2_in_gradient_l2_outgrad[0][0].png");
                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv1_in_gradient_l1_outgrad[0][0])), "outputs/tconv1_in_gradient_l1_outgrad[0][0].png");
            }
        }
    }

    public static void main(String[] args) {
        // 2:15 min per epoch for batch size 8
        Generator_Implementation_Without_Batchnorm generator = null; // (Generator_Implementation_Without_Batchnorm) loadObject("models/gen_no_batchnorm.ser");

        if (generator == null)
            generator = new Generator_Implementation_Without_Batchnorm(8, new AdamHyperparameters(0.001, 0.9, 0.999, 0.00000001));

        System.out.println("tconv1 output shape : " + generator.tconv1.outputDepth + " " + generator.tconv1.outputHeight + " " + generator.tconv1.outputWidth);
        System.out.println("tconv2 output shape : " + generator.tconv2.outputDepth + " " + generator.tconv2.outputHeight + " " + generator.tconv2.outputWidth);
        System.out.println("tconv3 output shape : " + generator.tconv3.outputDepth + " " + generator.tconv3.outputHeight + " " + generator.tconv3.outputWidth);
//        System.exit(0);

        // loading the first handwritten three from the mnist dataset
        BufferedImage img = MiscUtils.mnist_load_index(9, 0);

        double[][][] targetOutput = new double[][][]{MiscUtils.zeroToOneToMinusOneToOne(MiscUtils.img_to_mat(img))};


        double[][][][] outputGradients = new double[generator.batchSize][1][targetOutput[0].length][targetOutput[0][0].length];

        double prev_loss = Double.MAX_VALUE, loss = Double.MAX_VALUE;
        generator.verbose = true;

        for (int epoch = 0, max_epochs = 20000000; epoch < max_epochs && loss > 0.00001; epoch++, prev_loss = loss) {
            double[][][][] outputs = generator.forwardBatch();
            saveImage(getBufferedImage(outputs[0]), "outputs/gen_no_batchnorm_current.png");

            double[] losses = new double[generator.batchSize];
            for (int i = 0; i < generator.batchSize; i++)
                losses[i] = lossRMSE(outputs[i], targetOutput);
            loss = mean(losses);

            System.out.println("epoch : " + epoch + " loss : " + loss);

            for (int i = 0; i < generator.batchSize; i++)
                calculateGradientRMSE(outputGradients[i], outputs[i], targetOutput);

            generator.updateParametersBatch(outputGradients);

            if (epoch % 5 == 0) {
                saveObject(generator, "models/gen_no_batchnorm.ser");
            }
        }
    }
}