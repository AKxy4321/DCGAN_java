package DCGAN.networks;

import DCGAN.UTIL;
import DCGAN.XavierInitializer;
import DCGAN.layers.*;

import java.awt.image.BufferedImage;
import java.util.Arrays;

public class Generator_Implementation_single_strides {
    int dense_output_size;
    DenseLayer dense;
    BatchNormalization batch1;
    SigmoidLayer leakyReLU1;
    TransposeConvolutionalLayer tconv1;
    BatchNormalization batch2;
    LeakyReLULayer leakyReLU2;
    TransposeConvolutionalLayer tconv2;
    BatchNormalization batch3;
    LeakyReLULayer leakyReLU3;
    TransposeConvolutionalLayer tconv3;
    TanhLayer tanh;

    public boolean verbose = false;
    public int batchSize = 1;
    public int noise_length = 100;

    public Generator_Implementation_single_strides() {
        this(1);
    }

    public Generator_Implementation_single_strides(int batchSize) {
        this.batchSize = batchSize;

        this.noise_length = 100;
        int tconv1_input_width = 22, tconv1_input_height = 22, tconv1_input_depth = 73;
        this.dense_output_size = tconv1_input_width * tconv1_input_height * tconv1_input_depth;
        this.dense = new DenseLayer(noise_length, this.dense_output_size);
        this.batch1 = new BatchNormalization(this.dense_output_size);
        this.leakyReLU1 = new SigmoidLayer();

        this.tconv1 = new TransposeConvolutionalLayer(3, 127, 1,
                tconv1_input_width, tconv1_input_height, tconv1_input_depth, 0, false);
        this.batch2 = new BatchNormalization(tconv1.outputDepth * tconv1.outputHeight * tconv1.outputWidth);
        this.leakyReLU2 = new LeakyReLULayer();

        this.tconv2 = new TransposeConvolutionalLayer(3, 63, 1,
                tconv1.outputWidth, tconv1.outputHeight, tconv1.outputDepth, 0, false);
        this.batch3 = new BatchNormalization(tconv2.outputDepth * tconv2.outputHeight * tconv2.outputWidth);
        this.leakyReLU3 = new LeakyReLULayer();

        System.out.println("tconv2 output shape : " + tconv2.outputDepth + " " + tconv2.outputHeight + " " + tconv2.outputWidth);
        this.tconv3 = new TransposeConvolutionalLayer(3, 1, 1,
                tconv2.outputWidth, tconv2.outputHeight, tconv2.outputDepth, 0, false);
        this.tanh = new TanhLayer();


        // intialize the arrays
        noises = new double[batchSize][noise_length];

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
        double[] noise = XavierInitializer.xavierInit1D(this.noise_length); // generate noise input that we want to pass to the generator

        double[] gen_dense_output = this.dense.forward(noise);
        double[] gen_batch1_output = this.batch1.getOutput(gen_dense_output);
        double[][][] gen_batch1_output_unflattened = UTIL.unflatten(gen_batch1_output, tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);
        double[][][] gen_leakyrelu_output1 = this.leakyReLU1.forward(gen_batch1_output_unflattened);

        double[][][] outputTconv1 = this.tconv1.forward(gen_leakyrelu_output1);
        double[] gen_batch2_output = this.batch2.getOutput(UTIL.flatten(outputTconv1));
        double[][][] gen_batch2_output_unflattened = UTIL.unflatten(gen_batch2_output, outputTconv1.length, outputTconv1[0].length, outputTconv1[0][0].length);
        double[][][] gen_leakyrelu_output2 = this.leakyReLU2.forward(gen_batch2_output_unflattened);

        double[][][] outputTconv2 = this.tconv2.forward(gen_leakyrelu_output2);
        double[] gen_batch3_output = this.batch3.getOutput(UTIL.flatten(outputTconv2));
        double[][][] gen_batch3_output_unflattened = UTIL.unflatten(gen_batch3_output, outputTconv2.length, outputTconv2[0].length, outputTconv2[0][0].length);
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

        for (int i = 0; i < batchSize; i++) {
            noises[i] = XavierInitializer.xavierInit1D(100); // generate noise input that we want to pass to the generator
//            Arrays.fill(noises[i],0.5);
            gen_dense_outputs[i] = this.dense.forward(noises[i]);
        }

        gen_batchnorm1_outputs = this.batch1.forwardBatch(gen_dense_outputs);

        double[][] gen_outputs_tconv1_flattened = new double[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            double[][][] gen_batch1_output_unflattened = UTIL.unflatten(gen_batchnorm1_outputs[i], tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);
            gen_leakyrelu1_outputs[i] = this.leakyReLU1.forward(gen_batch1_output_unflattened);
            gen_tconv1_outputs[i] = this.tconv1.forward(gen_leakyrelu1_outputs[i]);

            gen_outputs_tconv1_flattened[i] = UTIL.flatten(gen_tconv1_outputs[i]);
        }
        gen_batchnorm2_outputs = this.batch2.forwardBatch(gen_outputs_tconv1_flattened);

        double[][] gen_outputs_tconv2_flattened = new double[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            double[][][] gen_batch2_output_unflattened = UTIL.unflatten(gen_batchnorm2_outputs[i], tconv2.inputDepth, tconv2.inputHeight, tconv2.inputWidth);
            gen_leakyrelu2_outputs[i] = this.leakyReLU2.forward(gen_batch2_output_unflattened);
            gen_tconv2_outputs[i] = this.tconv2.forward(gen_leakyrelu2_outputs[i]);

            gen_outputs_tconv2_flattened[i] = UTIL.flatten(gen_tconv2_outputs[i]);
        }

        gen_batchnorm3_outputs = this.batch3.forwardBatch(gen_outputs_tconv2_flattened);

        for (int i = 0; i < batchSize; i++) {
            double[][][] gen_batch3_output_unflattened = UTIL.unflatten(gen_batchnorm3_outputs[i], tconv3.inputDepth, tconv3.inputHeight, tconv3.inputWidth);
            gen_leakyrelu3_outputs[i] = this.leakyReLU3.forward(gen_batch3_output_unflattened);

            gen_tconv3_outputs[i] = this.tconv3.forward(gen_leakyrelu3_outputs[i]);

            double[][][] fakeImage = this.tanh.forward(gen_tconv3_outputs[i]);

            fakeImages[i] = fakeImage;
        }

        return fakeImages;
    }


    public void updateParametersBatch(double[][][][] gen_output_gradients, double learning_rate_gen) {
        double[][][][] tanh_in_gradients = new double[batchSize][][][];
        double[][] leakyrelu3_in_gradients_flattened = new double[batchSize][];
        double[][][][] leakyrelu3_in_gradients = new double[batchSize][][][];
        double[][][][] leakyrelu2_in_gradients = new double[batchSize][][][];
        double[][][][] leakyrelu_in_gradients = new double[batchSize][][][];
        double[][][][] batch3_in_gradients_unflattened = new double[batchSize][][][];
        double[][] leakyrelu2_in_gradients_flattened = new double[batchSize][];
        double[][][][] batch2_in_gradients_unflattened = new double[batchSize][][][];
        double[][] leakyrelu_in_gradients_flattened = new double[batchSize][];


        for (int i = 0; i < batchSize; i++) {
            tanh_in_gradients[i] = this.tanh.backward(gen_output_gradients[i]);

            leakyrelu3_in_gradients[i] = this.leakyReLU3.backward(this.tconv3.backward(tanh_in_gradients[i]));

            leakyrelu3_in_gradients_flattened[i] = UTIL.flatten(leakyrelu3_in_gradients[i]);
        }

        double[][] batch3_in_gradients = this.batch3.backward(leakyrelu3_in_gradients_flattened);

        for (int i = 0; i < batchSize; i++) {
            batch3_in_gradients_unflattened[i] = UTIL.unflatten(batch3_in_gradients[i], tconv2.outputDepth, tconv2.outputHeight, tconv2.outputWidth);
            double[][][] tconv2_in_gradient = this.tconv2.backward(batch3_in_gradients_unflattened[i]);

            leakyrelu2_in_gradients[i] = this.leakyReLU2.backward(tconv2_in_gradient);
            leakyrelu2_in_gradients_flattened[i] = UTIL.flatten(leakyrelu2_in_gradients[i]);
        }

        double[][] batch2_in_gradients_flattened = this.batch2.backward(leakyrelu2_in_gradients_flattened);


        for (int i = 0; i < batchSize; i++) {
            batch2_in_gradients_unflattened[i] = UTIL.unflatten(batch2_in_gradients_flattened[i], tconv1.outputDepth, tconv1.outputHeight, tconv1.outputWidth);

            double[][][] tconv1_in_gradient = this.tconv1.backward(batch2_in_gradients_unflattened[i]);

            leakyrelu_in_gradients[i] = this.leakyReLU1.backward(tconv1_in_gradient);
            leakyrelu_in_gradients_flattened[i] = UTIL.flatten(leakyrelu_in_gradients[i]);
        }

        double[][] batch1_in_gradients = this.batch1.backward(leakyrelu_in_gradients_flattened);

//        double[] dense_in_gradient = this.dense.backward(batch1_in_gradients[i]);


        double[] mean_batch1_in_gradients = UTIL.mean_1st_layer(batch1_in_gradients);
        double[][][] mean_batch2_in_gradients_unflattened = UTIL.mean_1st_layer(batch2_in_gradients_unflattened);
        double[][][] mean_batch3_in_gradients_unflattened = UTIL.mean_1st_layer(batch3_in_gradients_unflattened);
        double[][][] tanh_in_gradients_mean = UTIL.mean_1st_layer(tanh_in_gradients);

        for (int i = 0; i < batchSize; i++) {
            this.dense.updateParameters(mean_batch1_in_gradients, noises[i], learning_rate_gen);

            this.tconv1.updateParameters(mean_batch2_in_gradients_unflattened, gen_leakyrelu1_outputs[i], learning_rate_gen);

            this.tconv2.updateParameters(mean_batch3_in_gradients_unflattened, gen_leakyrelu2_outputs[i], learning_rate_gen);

            this.tconv3.updateParameters(tanh_in_gradients_mean, gen_leakyrelu3_outputs[i], learning_rate_gen);
        }

        this.batch1.updateParameters(leakyrelu_in_gradients_flattened, learning_rate_gen);
        this.batch2.updateParameters(leakyrelu2_in_gradients_flattened, learning_rate_gen);
        this.batch3.updateParameters(leakyrelu3_in_gradients_flattened, learning_rate_gen);


        if (verbose) {
            // print out the sum of each gradient by flattening it and summing it up using stream().sum()
            System.out.println("Sum of each gradient in generator: ");
            System.out.println("mean_batch1_in_gradients: " + Arrays.stream(mean_batch1_in_gradients).sum());
            System.out.println("mean_batch2_in_gradients_unflattened: " + Arrays.stream(UTIL.flatten(mean_batch2_in_gradients_unflattened)).sum());
            System.out.println("mean_batch3_in_gradients_unflattened: " + Arrays.stream(UTIL.flatten(mean_batch3_in_gradients_unflattened)).sum());
            System.out.println("tanh_in_gradients_mean: " + Arrays.stream(UTIL.flatten(tanh_in_gradients_mean)).sum());
            System.out.println("leakyrelu_in_gradients_flattened: " + Arrays.stream(UTIL.flatten(leakyrelu_in_gradients_flattened)).sum());
            System.out.println("leakyrelu2_in_gradients_flattened: " + Arrays.stream(UTIL.flatten(leakyrelu2_in_gradients_flattened)).sum());
            System.out.println("leakyrelu3_in_gradients_flattened: " + Arrays.stream(UTIL.flatten(leakyrelu3_in_gradients_flattened)).sum());
        }
    }

    public void updateParameters(double[][][] gen_output_gradient, double learning_rate_gen) {

        double[][][] tanh_in_gradient = this.tanh.backward(gen_output_gradient);
        double[][][] tconv3_in_gradient = this.tconv3.backward(tanh_in_gradient);
        double[][][] leakyrelu3_in_gradient = this.leakyReLU3.backward(tconv3_in_gradient);

//        double[] leakyrelu3_in_gradient_flattened = UTIL.flatten(leakyrelu3_in_gradient);
//        double[][][] batch3_in_gradient_unflattened = UTIL.unflatten(this.batch3.backward(leakyrelu3_in_gradient_flattened),
//                leakyrelu3_in_gradient.length, leakyrelu3_in_gradient[0].length, leakyrelu3_in_gradient[0][0].length);

        double[][][] tconv2_in_gradient = this.tconv2.backward(leakyrelu3_in_gradient);

        double[][][] leakyrelu2_in_gradient = this.leakyReLU2.backward(tconv2_in_gradient);
//        double[] leakyrelu2_in_gradient_flattened = UTIL.flatten(leakyrelu2_in_gradient);
//        double[][][] batch2_in_gradient_unflattened = UTIL.unflatten(
//                this.batch2.backward(leakyrelu2_in_gradient_flattened),
//                leakyrelu2_in_gradient.length, leakyrelu2_in_gradient[0].length, leakyrelu2_in_gradient[0][0].length);


        double[][][] tconv1_in_gradient = this.tconv1.backward(leakyrelu2_in_gradient);


        double[][][] leakyrelu_in_gradient = this.leakyReLU1.backward(tconv1_in_gradient);
        double[] leakyrelu_in_gradient_flattened = UTIL.flatten(leakyrelu_in_gradient);
//        double[] batch1_in_gradient = this.batch1.backward(leakyrelu_in_gradient_flattened);
//        gradient3 = UTIL.multiplyScalar(gradient3, 0.000001);

        double[] dense_in_gradient = this.dense.backward(leakyrelu_in_gradient_flattened);

        this.tconv1.updateParameters(leakyrelu2_in_gradient, learning_rate_gen);
        this.tconv2.updateParameters(leakyrelu3_in_gradient, learning_rate_gen);
//        this.batch1.updateParameters(leakyrelu_in_gradient_flattened, learning_rate_gen);
//        this.batch2.updateParameters(leakyrelu2_in_gradient_flattened, learning_rate_gen);
//        this.batch3.updateParameters(leakyrelu3_in_gradient_flattened, learning_rate_gen);
        this.dense.updateParameters(leakyrelu_in_gradient_flattened, learning_rate_gen);

        if (verbose) {
            // print out the sum of each gradient by flattening it and summing it up using stream().sum()
            System.out.println("Sum of each gradient in generator: ");

            System.out.println("gradient0: " + Arrays.stream(UTIL.flatten(tanh_in_gradient)).sum());
            System.out.println("gradient0_1: " + Arrays.stream(UTIL.flatten(leakyrelu3_in_gradient)).sum());
//            System.out.println("gradient0_2: " + Arrays.stream(UTIL.flatten(batch3_in_gradient_unflattened)).sum());
            System.out.println("gradient1: " + Arrays.stream(UTIL.flatten(tconv2_in_gradient)).sum());
            System.out.println("gradient1_2: " + Arrays.stream(UTIL.flatten(leakyrelu2_in_gradient)).sum());
//            System.out.println("gradient1_3: " + Arrays.stream(UTIL.flatten(batch2_in_gradient_unflattened)).sum());
            System.out.println("gradient2: " + Arrays.stream(UTIL.flatten(tconv1_in_gradient)).sum());
            System.out.println("gradient2_2: " + Arrays.stream(UTIL.flatten(leakyrelu_in_gradient)).sum());
            System.out.println("out: " + Arrays.stream(leakyrelu_in_gradient_flattened).sum());
//            System.out.println("gradient3: " + Arrays.stream(batch1_in_gradient).sum());
            System.out.println("gradient4: " + Arrays.stream(dense_in_gradient).sum());
        }
    }


    public static void main(String[] args) {
        Generator_Implementation_single_strides generator = new Generator_Implementation_single_strides(8);

        // loading the first handwritten three from the mnist dataset
        BufferedImage img = UTIL.mnist_load_index(3, 0);

        double[][][] targetOutput = new double[][][]{UTIL.zeroToOneToMinusOneToOne(UTIL.img_to_mat(img))};

        double[][][][] outputGradients = new double[generator.batchSize][1][28][28];

        double prev_loss = Double.MAX_VALUE, loss, learning_rate = 0.01;
        generator.verbose = true;

        for (int epoch = 0, max_epochs = 20000000; epoch < max_epochs; epoch++, prev_loss = loss) {
            double[][][][] outputs = generator.forwardBatch();
            UTIL.saveImage(UTIL.getBufferedImage(generator.generateImage()),"starting_image.png");

            double[] losses = new double[generator.batchSize];
            for (int i = 0; i < generator.batchSize; i++)
                losses[i] = UTIL.lossRMSE(outputs[i], targetOutput);
            loss = UTIL.mean(losses);

            System.out.println("loss : " + loss);


            for (int i = 0; i < generator.batchSize; i++)
                UTIL.calculateGradientRMSE(outputGradients[i], outputs[i], targetOutput);

            generator.updateParametersBatch(outputGradients, learning_rate);

            if (epoch % 10 == 0)
                UTIL.saveImage(UTIL.getBufferedImage(generator.generateImage()), "current_image_main_single_strides.png");

            if (loss < 0.1) break;
        }

        double[][][] output = generator.generateImage();
        UTIL.saveImage(UTIL.getBufferedImage(output), "final_image.png");
    }
}
/*
* // calculate the new learning rate based on difference in prev and curr loss
            double learning_rate = default_rate;
            if (epoch > 10) {
                double change_in_loss = prev_loss - loss;
                if (change_in_loss > 0)
                    learning_rate = largest / change_in_loss;
            }

            learning_rate = UTIL.clamp(learning_rate, smallest, largest);
            System.out.println("loss : " + loss + " learning_rate : " + default_rate);
* */