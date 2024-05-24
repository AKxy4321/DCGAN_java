package DCGAN.networks;

import DCGAN.util.MiscUtils;
import DCGAN.util.XavierInitializer;
import DCGAN.layers.*;

import java.awt.image.BufferedImage;
import java.util.Arrays;

import static DCGAN.util.MiscUtils.*;
import static DCGAN.util.TrainingUtils.calculateGradientRMSE;
import static DCGAN.util.TrainingUtils.lossRMSE;

public class Generator_Implementation_Without_Batchnorm {
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
    public int batchSize = 1;

    public Generator_Implementation_Without_Batchnorm(int batchSize, double learning_rate) {
        this.batchSize = batchSize;

        int noise_length = 100;
        int tconv1_input_width = 4, tconv1_input_height = 4, tconv1_input_depth = 256;
        this.dense_output_size = tconv1_input_width * tconv1_input_height * tconv1_input_depth;
        this.dense = new DenseLayer(noise_length, this.dense_output_size, learning_rate);
        this.leakyReLU1 = new LeakyReLULayer(0);

        // this.stride * (inputHeight - 1) + filterSize - 2 * padding;
        this.tconv1 = new TransposeConvolutionalLayer(3, 128, 2,
                tconv1_input_width, tconv1_input_height, tconv1_input_depth,
                1, 0, 0, 1, false, learning_rate);
        tconv1.outputHeight = 7;
        tconv1.outputWidth = 7;
        this.leakyReLU2 = new LeakyReLULayer(0);

        this.tconv2 = new TransposeConvolutionalLayer(4, 64, 2,
                tconv1.outputWidth, tconv1.outputHeight, tconv1.outputDepth,
                2, 0, 0, 1, false, learning_rate);
        tconv2.outputHeight = 14;
        tconv2.outputWidth = 14;
        this.leakyReLU3 = new LeakyReLULayer(0);

        this.tconv3 = new TransposeConvolutionalLayer(4, 1, 2,
                tconv2.outputWidth, tconv2.outputHeight, tconv2.outputDepth,
                2, 0, 0, 1, false, learning_rate);
        tconv3.outputHeight = 28;
        tconv3.outputWidth = 28;
        // (i + 4 - 1) + 3 - 2*0 = i + 6
        this.tanh = new TanhLayer();


        noises = new double[batchSize][dense.inputSize];
        denseOutputs = new double[batchSize][];
        denseOutputsUnflattened = new double[batchSize][][][];
        leakyReLU1Outputs = new double[batchSize][][][];
        tconv1Outputs = new double[batchSize][][][];
        leakyReLU2Outputs = new double[batchSize][][][];
        tconv2Outputs = new double[batchSize][][][];
        leakyReLU3Outputs = new double[batchSize][][][];
        tconv3Outputs = new double[batchSize][][][];
        tanhOutputs = new double[batchSize][][][];
    }

    public double[][][] generateImage() {
        double[] noise = XavierInitializer.xavierInit1D(500); // generate noise input that we want to pass to the generator

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
        for (int i = 0; i < batchSize; i++,  System.out.print(verbose ? " " + i : "")) {
            noises[i] = XavierInitializer.xavierInit1D(dense.inputSize);

            denseOutputs[i] = dense.forward(noises[i]);

            denseOutputsUnflattened[i] = MiscUtils.unflatten(denseOutputs[i], tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);

            leakyReLU1Outputs[i] = leakyReLU1.forward(denseOutputsUnflattened[i]);

            tconv1Outputs[i] = tconv1.forward(leakyReLU1Outputs[i]);

            leakyReLU2Outputs[i] = leakyReLU2.forward(tconv1Outputs[i]);

            tconv2Outputs[i] = tconv2.forward(leakyReLU2Outputs[i]);

            leakyReLU3Outputs[i] = leakyReLU3.forward(tconv2Outputs[i]);

            tconv3Outputs[i] = tconv3.forward(leakyReLU3Outputs[i]);

            tanhOutputs[i] = tanh.forward(tconv3Outputs[i]);
        }

        if(verbose){
            MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv1Outputs[0][0])), "tconv1Outputs[0][0].png");
            MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv2Outputs[0][0])), "tconv2Outputs[0][0].png");
            MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv3Outputs[0][0])), "tconv3Outputs[0][0].png");
        }


        return tanhOutputs;
    }

    public void updateParameters(double[][][] gen_output_gradient) {

        double[][][] tanh_in_gradient_t3_outgrad = this.tanh.backward(gen_output_gradient);
        double[][][] tconv3_in_gradient_l3_outgrad = this.tconv3.backward(tanh_in_gradient_t3_outgrad);
        double[][][] leakyrelu3_in_gradient_t2_outgrad = this.leakyReLU3.backward(tconv3_in_gradient_l3_outgrad);

        double[][][] tconv2_in_gradient_l2_outgrad = this.tconv2.backward(leakyrelu3_in_gradient_t2_outgrad);
        double[][][] leakyrelu2_in_gradient_t1_outgrad = this.leakyReLU2.backward(tconv2_in_gradient_l2_outgrad);

        double[][][] tconv1_in_gradient_l1_outgrad = this.tconv1.backward(leakyrelu2_in_gradient_t1_outgrad);
        double[][][] leakyrelu_in_gradient_d_outgrad = this.leakyReLU1.backward(tconv1_in_gradient_l1_outgrad);

        double[] leakyrelu_in_gradient_d_outgrad_flattened = MiscUtils.flatten(leakyrelu_in_gradient_d_outgrad);

        double[] dense_in_gradient = this.dense.backward(leakyrelu_in_gradient_d_outgrad_flattened);

        this.tconv1.updateParameters(leakyrelu2_in_gradient_t1_outgrad);
        this.tconv2.updateParameters(leakyrelu3_in_gradient_t2_outgrad);
        this.tconv3.updateParameters(tanh_in_gradient_t3_outgrad);
        this.dense.updateParameters(leakyrelu_in_gradient_d_outgrad_flattened);

        if (verbose) {
            // print out the sum of each gradient by flattening it and summing it up using stream().sum()
            System.out.println("Sum of each gradient in generator: ");

            System.out.println("tanh_in_gradient: " + Arrays.stream(MiscUtils.flatten(tanh_in_gradient_t3_outgrad)).sum());
            System.out.println("leakyrelu3_in_gradient: " + Arrays.stream(MiscUtils.flatten(leakyrelu3_in_gradient_t2_outgrad)).sum());
            System.out.println("tconv2_in_gradient: " + Arrays.stream(MiscUtils.flatten(tconv2_in_gradient_l2_outgrad)).sum());
            System.out.println("leakyrelu2_in_gradient: " + Arrays.stream(MiscUtils.flatten(leakyrelu2_in_gradient_t1_outgrad)).sum());
            System.out.println("tconv1_in_gradient: " + Arrays.stream(MiscUtils.flatten(tconv1_in_gradient_l1_outgrad)).sum());
            System.out.println("leakyrelu_in_gradient: " + Arrays.stream(MiscUtils.flatten(leakyrelu_in_gradient_d_outgrad)).sum());
            System.out.println("leakyrelu_in_gradient_flattened: " + Arrays.stream(leakyrelu_in_gradient_d_outgrad_flattened).sum());
            System.out.println("dense_in_gradient: " + Arrays.stream(dense_in_gradient).sum());
        }
    }

    public void updateParametersBatch(double[][][][] outputGradients) {
        double[][][][] tanh_in_gradient_t3_outgrad = new double[batchSize][][][];
        double[][][][] tconv3_in_gradient_l3_outgrad = new double[batchSize][][][];
        double[][][][] leakyrelu3_in_gradient_t2_outgrad = new double[batchSize][][][];
        double[][][][] tconv2_in_gradient_l2_outgrad = new double[batchSize][][][];
        double[][][][] leakyrelu2_in_gradient_t1_outgrad = new double[batchSize][][][];
        double[][][][] tconv1_in_gradient_l1_outgrad = new double[batchSize][][][];
        double[][] leakyrelu_in_gradient_d_outgrad_flattened = new double[batchSize][];

        for (int i = 0; i < batchSize; i++,  System.out.print(verbose ? " " + i : "")) {
            tanh_in_gradient_t3_outgrad[i] = this.tanh.backward(outputGradients[i], tanhOutputs[i]);
            tconv3_in_gradient_l3_outgrad[i] = this.tconv3.backward(tanh_in_gradient_t3_outgrad[i]);
            leakyrelu3_in_gradient_t2_outgrad[i] = this.leakyReLU3.backward(tconv3_in_gradient_l3_outgrad[i], leakyReLU3Outputs[i]);

            tconv2_in_gradient_l2_outgrad[i] = this.tconv2.backward(leakyrelu3_in_gradient_t2_outgrad[i]);
            leakyrelu2_in_gradient_t1_outgrad[i] = this.leakyReLU2.backward(tconv2_in_gradient_l2_outgrad[i], leakyReLU2Outputs[i]);

            tconv1_in_gradient_l1_outgrad[i] = this.tconv1.backward(leakyrelu2_in_gradient_t1_outgrad[i]);
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

                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tanh_in_gradient_t3_outgrad[0][0])), "tanh_in_gradient_t3_outgrad[0][0].png");
                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv3_in_gradient_l3_outgrad[0][0])), "tconv3_in_gradient_l3_outgrad[0][0].png");
                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv2_in_gradient_l2_outgrad[0][0])), "tconv2_in_gradient_l2_outgrad[0][0].png");
                MiscUtils.saveImage(getBufferedImage(scaleMinMax(tconv1_in_gradient_l1_outgrad[0][0])), "tconv1_in_gradient_l1_outgrad[0][0].png");
            }
        }
    }

    public static void main(String[] args) {
        Generator_Implementation_Without_Batchnorm generator = new Generator_Implementation_Without_Batchnorm(8, 0.001);

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
            saveImage(getBufferedImage(outputs[0]), "gen_no_batchnorm_current.png");

            double[] losses = new double[generator.batchSize];
            for (int i = 0; i < generator.batchSize; i++)
                losses[i] = lossRMSE(outputs[i], targetOutput);
            loss = mean(losses);

            System.out.println("epoch : " + epoch + " loss : " + loss);

            for (int i = 0; i < generator.batchSize; i++)
                calculateGradientRMSE(outputGradients[i], outputs[i], targetOutput);

            generator.updateParametersBatch(outputGradients);
        }
    }
}