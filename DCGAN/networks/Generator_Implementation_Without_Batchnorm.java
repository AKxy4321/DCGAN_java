package DCGAN.networks;

import DCGAN.UTIL;
import DCGAN.XavierInitializer;
import DCGAN.layers.*;

import java.awt.image.BufferedImage;
import java.util.Arrays;

public class Generator_Implementation_Without_Batchnorm {
    int dense_output_size;
    DenseLayer dense;
    SigmoidLayer leakyReLU1;
    TransposeConvolutionalLayer tconv1;
    LeakyReLULayer leakyReLU2;
    TransposeConvolutionalLayer tconv2;
    LeakyReLULayer leakyReLU3;
    TransposeConvolutionalLayer tconv3;
    TanhLayer tanh;

    public boolean verbose = false;
    public int batchSize = 1;

    public Generator_Implementation_Without_Batchnorm() {
        this(1);
    }

    public Generator_Implementation_Without_Batchnorm(int batchSize) {
        this.batchSize = batchSize;

        int noise_length = 500;
        int tconv1_input_width = 22, tconv1_input_height = 22, tconv1_input_depth = 127;
        this.dense_output_size = tconv1_input_width * tconv1_input_height * tconv1_input_depth;
        this.dense = new DenseLayer(noise_length, this.dense_output_size);
        this.leakyReLU1 = new SigmoidLayer();

        this.tconv1 = new TransposeConvolutionalLayer(3, 33, 1,
                tconv1_input_width, tconv1_input_height, tconv1_input_depth, 0, false);
        // this.stride * (inputHeight - 1) + filterSize - 2 * padding; = 1 * (7 - 1) + 5 - 0 = 11
        // 1 * (i-1) + 3 - 0 = i + 2
        this.leakyReLU2 = new LeakyReLULayer();

        this.tconv2 = new TransposeConvolutionalLayer(3, 11, 1,
                tconv1.outputWidth, tconv1.outputHeight, tconv1.outputDepth, 0, false);
        // (i + 2 - 1) + 3 - 2*0 = i + 4
        this.leakyReLU3 = new LeakyReLULayer();

        System.out.println("tconv2 output shape : " + tconv2.outputDepth + " " + tconv2.outputHeight + " " + tconv2.outputWidth);
        this.tconv3 = new TransposeConvolutionalLayer(3, 1, 1,
                tconv2.outputWidth, tconv2.outputHeight, tconv2.outputDepth, 0, false);
        // (i + 4 - 1) + 3 - 2*0 = i + 6
        this.tanh = new TanhLayer();
    }

    public double[][][] generateImage() {
        double[] noise = XavierInitializer.xavierInit1D(500); // generate noise input that we want to pass to the generator

        double[] gen_dense_output = this.dense.forward(noise);
        double[][][] gen_dense_output_unflattened = UTIL.unflatten(gen_dense_output, tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);
        double[][][] gen_leakyrelu_output1 = this.leakyReLU1.forward(gen_dense_output_unflattened);

        UTIL.saveImage(UTIL.getBufferedImage(gen_leakyrelu_output1), "gen_leakyrelu_output1.png");

        double[][][] outputTconv1 = this.tconv1.forward(gen_leakyrelu_output1);
        double[][][] gen_leakyrelu_output2 = this.leakyReLU2.forward(outputTconv1);

        UTIL.saveImage(UTIL.getBufferedImage(gen_leakyrelu_output2), "gen_leakyrelu_output2.png");

        double[][][] outputTconv2 = this.tconv2.forward(gen_leakyrelu_output2);
        double[][][] gen_leakyrelu_output3 = this.leakyReLU3.forward(outputTconv2);

        UTIL.saveImage(UTIL.getBufferedImage(gen_leakyrelu_output3), "gen_leakyrelu_output3.png");

        double[][][] gen_tconv3_output = this.tconv3.forward(gen_leakyrelu_output3);
        double[][][] fakeImage = this.tanh.forward(gen_tconv3_output);
        return fakeImage;
    }

    public void updateParameters(double[][][] gen_output_gradient, double learning_rate_gen) {

        double[][][] tanh_in_gradient_t3_outgrad = this.tanh.backward(gen_output_gradient);
        double[][][] tconv3_in_gradient_l3_outgrad = this.tconv3.backward(tanh_in_gradient_t3_outgrad);
        double[][][] leakyrelu3_in_gradient_t2_outgrad = this.leakyReLU3.backward(tconv3_in_gradient_l3_outgrad);

        double[][][] tconv2_in_gradient_l2_outgrad = this.tconv2.backward(leakyrelu3_in_gradient_t2_outgrad);
        double[][][] leakyrelu2_in_gradient_t1_outgrad = this.leakyReLU2.backward(tconv2_in_gradient_l2_outgrad);

        double[][][] tconv1_in_gradient_l1_outgrad = this.tconv1.backward(leakyrelu2_in_gradient_t1_outgrad);
        double[][][] leakyrelu_in_gradient_d_outgrad = this.leakyReLU1.backward(tconv1_in_gradient_l1_outgrad);

        double[] leakyrelu_in_gradient_d_outgrad_flattened = UTIL.flatten(leakyrelu_in_gradient_d_outgrad);

        double[] dense_in_gradient = this.dense.backward(leakyrelu_in_gradient_d_outgrad_flattened);

        this.tconv1.updateParameters(leakyrelu2_in_gradient_t1_outgrad, learning_rate_gen);
        this.tconv2.updateParameters(leakyrelu3_in_gradient_t2_outgrad, learning_rate_gen);
        this.tconv3.updateParameters(tanh_in_gradient_t3_outgrad, learning_rate_gen);
        this.dense.updateParameters(leakyrelu_in_gradient_d_outgrad_flattened, learning_rate_gen);

        if (verbose) {
            // print out the sum of each gradient by flattening it and summing it up using stream().sum()
            System.out.println("Sum of each gradient in generator: ");

            System.out.println("tanh_in_gradient: " + Arrays.stream(UTIL.flatten(tanh_in_gradient_t3_outgrad)).sum());
            System.out.println("leakyrelu3_in_gradient: " + Arrays.stream(UTIL.flatten(leakyrelu3_in_gradient_t2_outgrad)).sum());
            System.out.println("tconv2_in_gradient: " + Arrays.stream(UTIL.flatten(tconv2_in_gradient_l2_outgrad)).sum());
            System.out.println("leakyrelu2_in_gradient: " + Arrays.stream(UTIL.flatten(leakyrelu2_in_gradient_t1_outgrad)).sum());
            System.out.println("tconv1_in_gradient: " + Arrays.stream(UTIL.flatten(tconv1_in_gradient_l1_outgrad)).sum());
            System.out.println("leakyrelu_in_gradient: " + Arrays.stream(UTIL.flatten(leakyrelu_in_gradient_d_outgrad)).sum());
            System.out.println("leakyrelu_in_gradient_flattened: " + Arrays.stream(leakyrelu_in_gradient_d_outgrad_flattened).sum());
            System.out.println("dense_in_gradient: " + Arrays.stream(dense_in_gradient).sum());
        }
    }


    public static void main(String[] args) {
        Generator_Implementation_Without_Batchnorm generator = new Generator_Implementation_Without_Batchnorm(8);

        // loading the first handwritten three from the mnist dataset
        BufferedImage img = UTIL.mnist_load_index(3, 0);

        double[][][] targetOutput = new double[][][]{UTIL.zeroToOneToMinusOneToOne(UTIL.img_to_mat(img))};
//        targetOutput = UTIL.multiplyScalar(targetOutput, -1);
        UTIL.prettyprint(targetOutput);
        UTIL.saveImage(UTIL.getBufferedImage(targetOutput), "target_image.png");

        double[][][] outputGradients = new double[1][28][28];

        double prev_loss = Double.MAX_VALUE, loss, learning_rate = 0.1;
        generator.verbose = true;

        for (int epoch = 0, max_epochs = 20000000; epoch < max_epochs; epoch++, prev_loss = loss) {
            double[][][] output = generator.generateImage();

            UTIL.saveImage(UTIL.getBufferedImage(generator.generateImage()),"starting_image.png");

            loss = UTIL.lossRMSE(output, targetOutput);

            System.out.println("loss : " + loss);

            UTIL.calculateGradientRMSE(outputGradients, output, targetOutput);

//            outputGradients = UTIL.multiplyScalar(outputGradients, 28*28.0);

            generator.updateParameters(outputGradients, learning_rate);

            if (epoch % 10 == 0)
                UTIL.saveImage(UTIL.getBufferedImage(generator.generateImage()), "current_image.png");

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