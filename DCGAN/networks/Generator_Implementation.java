package DCGAN.networks;

import DCGAN.optimizers.AdamHyperparameters;
import DCGAN.optimizers.OptimizerHyperparameters;
import DCGAN.util.MiscUtils;
import DCGAN.util.ArrayInitializer;
import DCGAN.layers.*;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.Arrays;

import static DCGAN.util.MathUtils.mean;
import static DCGAN.util.MiscUtils.*;
import static DCGAN.util.TrainingUtils.*;

public class Generator_Implementation implements Serializable {
    private static final long serialVersionUID = 1L;
    /**
     * TODO: There is a (almost) vanishing gradient problem when backpropagating from the batch normalization layer. Have to debug.
     * But it does give some output close to the target image after an hour, so maybe only the layers after the last batch normalization layer is learning
     * debug output :
     * (epoch 205)
     * Sum of each gradient in generator:
     * dense_in_gradients: 9.367506770274758E-17
     * batch1_in_gradients: 9.367506770274758E-17
     * tconv1_in_gradients: 1.3931997916438732E-17
     * batch2_in_gradients: 1.3931997916438732E-17
     * tconv2_in_gradients: -1.5151725360484924E-17
     * batch3_in_gradients: -1.5151725360484924E-17
     * tconv3_in_gradients: -0.003934218516113986
     * tanh_in_gradients: -0.003934218516113986
     */

    int dense_output_size;
    DenseLayer dense;
    BatchNormalization batch1;
    LeakyReLULayer leakyReLU1;
    TransposeConvolutionalLayer tconv1;
    BatchNormalization batch2;
    LeakyReLULayer leakyReLU2;
    TransposeConvolutionalLayer tconv2;
    BatchNormalization batch3;
    LeakyReLULayer leakyReLU3;
    TransposeConvolutionalLayer tconv3;
    TanhLayer tanh;

    public boolean verbose = false;
    public int batchSize;
    int noise_length = 100;


    public Generator_Implementation(int batchSize, OptimizerHyperparameters optimizerHyperparameters) {
        this.batchSize = batchSize;

        int tconv1_input_width = 4, tconv1_input_height = 4, tconv1_input_depth = 256;
        this.dense_output_size = tconv1_input_width * tconv1_input_height * tconv1_input_depth;


        this.dense = new DenseLayer(noise_length, this.dense_output_size, optimizerHyperparameters);
        this.batch1 = new BatchNormalization(this.dense_output_size, optimizerHyperparameters);
        this.leakyReLU1 = new LeakyReLULayer();

        this.tconv1 = new TransposeConvolutionalLayer(3, 128, 2,
                tconv1_input_width, tconv1_input_height, tconv1_input_depth,
                1, 0, 0, 1, false, optimizerHyperparameters);
        assert tconv1.outputHeight == 7;
        assert tconv1.outputWidth == 7;
        this.batch2 = new BatchNormalization(tconv1.outputDepth * tconv1.outputHeight * tconv1.outputWidth, optimizerHyperparameters);
        this.leakyReLU2 = new LeakyReLULayer();

        this.tconv2 = new TransposeConvolutionalLayer(4, 64, 2,
                tconv1.outputWidth, tconv1.outputHeight, tconv1.outputDepth,
                2, 0, 0, 1, false, optimizerHyperparameters);
        assert tconv2.outputHeight == 14;
        assert tconv2.outputWidth == 14;
        this.batch3 = new BatchNormalization(tconv2.outputDepth * tconv2.outputHeight * tconv2.outputWidth, optimizerHyperparameters);
        this.leakyReLU3 = new LeakyReLULayer();

        this.tconv3 = new TransposeConvolutionalLayer(4, 1, 2,
                tconv2.outputWidth, tconv2.outputHeight, tconv2.outputDepth,
                2, 0, 0, 1, false, optimizerHyperparameters);
        assert tconv3.outputHeight == 28;
        assert tconv3.outputWidth == 28;
        this.tanh = new TanhLayer();


        initArrays();
    }

    private void initArrays() {
        /**
         * initializes the arrays that will be used during forward and backward passes of batches
         * */

        noises = new double[batchSize][100];

        gen_dense_outputs = new double[batchSize][dense_output_size];
        gen_batchnorm1_outputs = new double[batchSize][dense_output_size];
        gen_leakyrelu1_outputs = new double[batchSize][tconv1.inputDepth][tconv1.inputHeight][tconv1.inputWidth];

        gen_tconv1_outputs = new double[batchSize][tconv1.outputDepth][tconv1.outputHeight][tconv1.outputWidth];
        gen_leakyrelu2_outputs = new double[batchSize][tconv1.outputDepth][tconv1.outputHeight][tconv1.outputWidth];

        gen_batchnorm2_outputs = new double[batchSize][tconv1.outputDepth * tconv1.outputHeight * tconv1.outputWidth];

        gen_tconv2_outputs = new double[batchSize][tconv2.outputDepth][tconv2.outputHeight][tconv2.outputWidth];
        gen_leakyrelu3_outputs = new double[batchSize][tconv2.outputDepth][tconv2.outputHeight][tconv2.outputWidth];

        gen_batchnorm3_outputs = new double[batchSize][tconv2.outputDepth * tconv2.outputHeight * tconv2.outputWidth];
        gen_tconv3_outputs = new double[batchSize][tconv3.outputDepth][tconv3.outputHeight][tconv3.outputWidth];

        fakeImages = new double[batchSize][tconv3.outputDepth][tconv3.outputHeight][tconv3.outputWidth];
    }

    public double[][][] generateImage() {
        double[] noise = ArrayInitializer.xavierInit1D(noise_length); // generate noise input that we want to pass to the generator

        double[] gen_dense_output = this.dense.forward(noise);
        double[] gen_batch1_output = this.batch1.getOutput(gen_dense_output);
        double[][][] gen_batch1_output_unflattened = MiscUtils.unflatten(gen_batch1_output, tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);
        double[][][] gen_leakyrelu_output1 = this.leakyReLU1.forward(gen_batch1_output_unflattened);

        double[][][] outputTconv1 = this.tconv1.forward(gen_leakyrelu_output1);
        double[] gen_batch2_output = this.batch2.getOutput(MiscUtils.flatten(outputTconv1));
        double[][][] gen_batch2_output_unflattened = MiscUtils.unflatten(gen_batch2_output, outputTconv1.length, outputTconv1[0].length, outputTconv1[0][0].length);
        double[][][] gen_leakyrelu_output2 = this.leakyReLU2.forward(gen_batch2_output_unflattened);

        double[][][] outputTconv2 = this.tconv2.forward(gen_leakyrelu_output2);
        double[] gen_batch3_output = this.batch3.getOutput(MiscUtils.flatten(outputTconv2));
        double[][][] gen_batch3_output_unflattened = MiscUtils.unflatten(gen_batch3_output, outputTconv2.length, outputTconv2[0].length, outputTconv2[0][0].length);
        double[][][] gen_leakyrelu_output3 = this.leakyReLU3.forward(gen_batch3_output_unflattened);


        double[][][] gen_tconv3_output = this.tconv3.forward(gen_leakyrelu_output3);
        double[][][] fakeImage = this.tanh.forward(gen_tconv3_output);
        return fakeImage;
    }

    double[][] gen_dense_outputs;
    double[][] gen_batchnorm1_outputs;
    double[][][][] gen_leakyrelu1_outputs;
    double[][][][] gen_tconv1_outputs;
    double[][] gen_batchnorm2_outputs;
    double[][][][] gen_leakyrelu2_outputs;
    double[][][][] gen_tconv2_outputs;
    double[][] gen_batchnorm3_outputs;
    double[][][][] gen_leakyrelu3_outputs;
    double[][][][] gen_tconv3_outputs;
    double[][][][] fakeImages;
    double[][] noises;

    public double[][][][] forwardBatch() {
        // reset Arrays

        for (int i = 0; i < batchSize; i++) {
            noises[i] = ArrayInitializer.xavierInit1D(100); // generate noise input that we want to pass to the generator
            gen_dense_outputs[i] = this.dense.forward(noises[i]);
        }

        gen_batchnorm1_outputs = this.batch1.forwardBatch(gen_dense_outputs);

        double[][] gen_outputs_tconv1_flattened = new double[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            double[][][] gen_batch1_output_unflattened = MiscUtils.unflatten(gen_batchnorm1_outputs[i], tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);
            gen_leakyrelu1_outputs[i] = this.leakyReLU1.forward(gen_batch1_output_unflattened);
            this.tconv1.forward(gen_tconv1_outputs[i], gen_leakyrelu1_outputs[i]); // do forward pass and store the result in gen_tconv1_outputs[i]

            gen_outputs_tconv1_flattened[i] = MiscUtils.flatten(gen_tconv1_outputs[i]);
        }
        gen_batchnorm2_outputs = this.batch2.forwardBatch(gen_outputs_tconv1_flattened);

        double[][] gen_outputs_tconv2_flattened = new double[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            double[][][] gen_batch2_output_unflattened = MiscUtils.unflatten(gen_batchnorm2_outputs[i], tconv2.inputDepth, tconv2.inputHeight, tconv2.inputWidth);
            gen_leakyrelu2_outputs[i] = this.leakyReLU2.forward(gen_batch2_output_unflattened);
            this.tconv2.forward(gen_tconv2_outputs[i], gen_leakyrelu2_outputs[i]); // do forward pass and store the result in gen_tconv2_outputs[i]

            gen_outputs_tconv2_flattened[i] = MiscUtils.flatten(gen_tconv2_outputs[i]);
        }

        gen_batchnorm3_outputs = this.batch3.forwardBatch(gen_outputs_tconv2_flattened);

        for (int i = 0; i < batchSize; i++) {
            double[][][] gen_batch3_output_unflattened = MiscUtils.unflatten(gen_batchnorm3_outputs[i], tconv3.inputDepth, tconv3.inputHeight, tconv3.inputWidth);
            gen_leakyrelu3_outputs[i] = this.leakyReLU3.forward(gen_batch3_output_unflattened);

            this.tconv3.forward(gen_tconv3_outputs[i], gen_leakyrelu3_outputs[i]); // do forward pass and store the result in gen_tconv3_outputs[i]

            double[][][] fakeImage = this.tanh.forward(gen_tconv3_outputs[i]);

            fakeImages[i] = fakeImage;
        }

        return fakeImages;
    }


    public void updateParametersBatch(double[][][][] gen_output_gradients) {
        /**
         * updates the parameters without returning the gradient w.r.t input.
         * The input gradient for a layer is the output gradient of the previous layer, hence the _in_ in the name
         * */
        double[][][][] tanh_in_gradients = new double[batchSize][][][];
        double[][] leakyrelu3_in_gradients_flattened_batch3_outgrad = new double[batchSize][];
        double[][][][] leakyrelu3_in_gradients = new double[batchSize][][][];
        double[][][][] leakyrelu2_in_gradients = new double[batchSize][][][];
        double[][][][] leakyrelu_in_gradients = new double[batchSize][][][];
        double[][][][] batch3_in_gradients_unflattened = new double[batchSize][][][];
        double[][] leakyrelu2_in_gradients_flattened_batch2_outgrad = new double[batchSize][];
        double[][][][] batch2_in_gradients_unflattened = new double[batchSize][][][];
        double[][] leakyrelu_in_gradients_flattened_batch1_outgrad = new double[batchSize][];


        for (int i = 0; i < batchSize; i++, System.out.print(verbose ? i + " " : " ")) {
            tanh_in_gradients[i] = this.tanh.backward(gen_output_gradients[i], fakeImages[i]);

            leakyrelu3_in_gradients[i] = this.leakyReLU3.backward(this.tconv3.backward(tanh_in_gradients[i]), gen_leakyrelu3_outputs[i]);

            leakyrelu3_in_gradients_flattened_batch3_outgrad[i] = MiscUtils.flatten(leakyrelu3_in_gradients[i]);
        }

        double[][] batch3_in_gradients = this.batch3.backward(leakyrelu3_in_gradients_flattened_batch3_outgrad);

        for (int i = 0; i < batchSize; i++, System.out.print(verbose ? i + " " : " ")) {
            batch3_in_gradients_unflattened[i] = MiscUtils.unflatten(batch3_in_gradients[i], tconv2.outputDepth, tconv2.outputHeight, tconv2.outputWidth);
            double[][][] tconv2_in_gradient = this.tconv2.backward(batch3_in_gradients_unflattened[i]);

            leakyrelu2_in_gradients[i] = this.leakyReLU2.backward(tconv2_in_gradient, gen_leakyrelu2_outputs[i]);
            leakyrelu2_in_gradients_flattened_batch2_outgrad[i] = MiscUtils.flatten(leakyrelu2_in_gradients[i]);
        }

        double[][] batch2_in_gradients_flattened = this.batch2.backward(leakyrelu2_in_gradients_flattened_batch2_outgrad);

        for (int i = 0; i < batchSize; i++, System.out.print(verbose ? i + " " : " ")) {
            batch2_in_gradients_unflattened[i] = MiscUtils.unflatten(batch2_in_gradients_flattened[i], tconv1.outputDepth, tconv1.outputHeight, tconv1.outputWidth);

            double[][][] tconv1_in_gradient = this.tconv1.backward(batch2_in_gradients_unflattened[i]);

            leakyrelu_in_gradients[i] = this.leakyReLU1.backward(tconv1_in_gradient, gen_leakyrelu1_outputs[i]);
            leakyrelu_in_gradients_flattened_batch1_outgrad[i] = MiscUtils.flatten(leakyrelu_in_gradients[i]);
        }

        double[][] batch1_in_gradients = this.batch1.backward(leakyrelu_in_gradients_flattened_batch1_outgrad);


        this.dense.updateParametersBatch(batch1_in_gradients, noises);
        this.batch1.updateParameters(leakyrelu_in_gradients_flattened_batch1_outgrad);

        this.tconv1.updateParametersBatch(batch2_in_gradients_unflattened, gen_leakyrelu1_outputs);
        this.batch2.updateParameters(leakyrelu2_in_gradients_flattened_batch2_outgrad);

        this.tconv2.updateParametersBatch(batch3_in_gradients_unflattened, gen_leakyrelu2_outputs);
        this.batch3.updateParameters(leakyrelu3_in_gradients_flattened_batch3_outgrad);

        this.tconv3.updateParametersBatch(tanh_in_gradients, gen_leakyrelu3_outputs);


        if (verbose) {
            // print out the sum of each gradient by flattening it and summing it up using stream().sum()
            System.out.println("Sum of each gradient in generator: ");
            System.out.println("dense_in_gradients: " + Arrays.stream(MiscUtils.flatten(batch1_in_gradients)).sum());
            System.out.println("batch1_in_gradients: " + Arrays.stream(MiscUtils.flatten(batch1_in_gradients)).sum());
            System.out.println("tconv1_in_gradients: " + Arrays.stream(MiscUtils.flatten(batch2_in_gradients_flattened)).sum());
            System.out.println("batch2_in_gradients: " + Arrays.stream(MiscUtils.flatten(batch2_in_gradients_flattened)).sum());
            System.out.println("tconv2_in_gradients: " + Arrays.stream(MiscUtils.flatten(batch3_in_gradients)).sum());
            System.out.println("batch3_in_gradients: " + Arrays.stream(MiscUtils.flatten(batch3_in_gradients)).sum());
            System.out.println("tconv3_in_gradients: " + Arrays.stream(MiscUtils.flatten(tanh_in_gradients[0])).sum());
            System.out.println("tanh_in_gradients: " + Arrays.stream(MiscUtils.flatten(tanh_in_gradients[0])).sum());

            MiscUtils.saveImage(getBufferedImage(scaleMinMax(leakyrelu3_in_gradients[0][0])), "leakyrelu3_in_gradients.png");
            MiscUtils.saveImage(getBufferedImage(scaleMinMax(batch3_in_gradients_unflattened[0][0])), "batch3_in_gradients_unflattened.png");
        }
    }

    public static void main(String[] args) {
        Generator_Implementation generator = new Generator_Implementation(3, new AdamHyperparameters(0.0002, 0.5, 0.999, 1e-8));
        generator.verbose = true;

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

        for (int epoch = 0, max_epochs = 20000000; epoch < max_epochs && loss > 0.1; epoch++, prev_loss = loss) {
            double[][][][] outputs = generator.forwardBatch();
            saveImage(getBufferedImage(outputs[0]), "gen_with_batchnorm_current.png");

            double[] losses = new double[generator.batchSize];
            for (int i = 0; i < generator.batchSize; i++)
                losses[i] = lossMSE(outputs[i], targetOutput);
            loss = mean(losses);

            System.out.println("epoch : " + epoch + " loss : " + loss);

            for (int i = 0; i < generator.batchSize; i++)
                calculateGradientMSE(outputGradients[i][0], outputs[i][0], targetOutput[0]);

            generator.updateParametersBatch(outputGradients);
        }
    }
}