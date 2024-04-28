package DCGAN;

import java.awt.image.BufferedImage;
import java.util.Arrays;

public class Generator_Implementation {
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

    public Generator_Implementation() {
        int noise_length = 100;
        int tconv1_input_width = 7, tconv1_input_height = 7, tconv1_input_depth = 5;
        this.dense_output_size = tconv1_input_width * tconv1_input_height * tconv1_input_depth;
        this.dense = new DenseLayer(noise_length, this.dense_output_size);
        this.batch1 = new BatchNormalization(this.dense_output_size);
        this.leakyReLU1 = new LeakyReLULayer();

        this.tconv1 = new TransposeConvolutionalLayer(5, 16, 1, tconv1_input_width, tconv1_input_height, tconv1_input_depth, 0);
        this.batch2 = new BatchNormalization(tconv1.outputDepth * tconv1.outputHeight * tconv1.outputWidth);
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(5, 16, 2, tconv1.outputWidth, tconv1.outputHeight, tconv1.outputDepth, 3);
        this.batch3 = new BatchNormalization(tconv2.outputDepth * tconv2.outputHeight * tconv2.outputWidth);
        this.leakyReLU3 = new LeakyReLULayer();
        this.tconv3 = new TransposeConvolutionalLayer(6, 1, 2, tconv2.outputWidth, tconv2.outputHeight, tconv2.outputDepth, 7);
        this.tanh = new TanhLayer();
    }

    public double[][][] generateImage() {
//        double[] noise = XavierInitializer.xavierInit1D(100); // generate noise input that we want to pass to the generator

        double[] noise = new double[100];
        Arrays.fill(noise,0.5);// TODO: go back to xavier Initialization

        double[] gen_dense_output = this.dense.forward(noise);
        double[] gen_batch1_output = this.batch1.forward(gen_dense_output, true);
        double[][][] gen_batch1_output_unflattened = UTIL.unflatten(gen_batch1_output, tconv1.inputDepth, tconv1.inputHeight, tconv1.inputWidth);
        double[][][] gen_leakyrelu_output1 = this.leakyReLU1.forward(gen_batch1_output_unflattened);
        double[][][] outputTconv1 = this.tconv1.forward(gen_leakyrelu_output1);
        double[] gen_batch2_output = this.batch2.forward(UTIL.flatten(outputTconv1), true);

        double[][][] gen_batch2_output_unflattened = UTIL.unflatten(gen_batch2_output, outputTconv1.length, outputTconv1[0].length, outputTconv1[0][0].length);
        double[][][] gen_leakyrelu_output2 = this.leakyReLU2.forward(gen_batch2_output_unflattened);
        double[][][] outputTconv2 = this.tconv2.forward(gen_leakyrelu_output2);
        double[] gen_batch3_output = this.batch3.forward(UTIL.flatten(outputTconv2), true);
        double[][][] gen_batch3_output_unflattened = UTIL.unflatten(gen_batch3_output, outputTconv2.length, outputTconv2[0].length, outputTconv2[0][0].length);
        double[][][] gen_leakyrelu_output3 = this.leakyReLU3.forward(gen_batch3_output_unflattened);

        double[][][] gen_tconv3_output = this.tconv3.forward(gen_leakyrelu_output3);

        double[][][] fakeImage = this.tanh.forward(gen_tconv3_output);
        return fakeImage;
    }

    public void updateParameters(double[][][] gen_output_gradient, double learning_rate_gen) {

        double[][][] tanh_in_gradient = this.tanh.backward(gen_output_gradient);
        double[][][] tconv3_in_gradient = this.tconv3.backward(tanh_in_gradient);
        double[][][] leakyrelu3_in_gradient = this.leakyReLU3.backward(tconv3_in_gradient);
        double[] leakyrelu3_in_gradient_flattened = UTIL.flatten(leakyrelu3_in_gradient);
        double[][][] batch3_in_gradient_unflattened = UTIL.unflatten(this.batch3.backward(leakyrelu3_in_gradient_flattened),
                leakyrelu3_in_gradient.length, leakyrelu3_in_gradient[0].length, leakyrelu3_in_gradient[0][0].length);

        double[][][] tconv2_in_gradient = this.tconv2.backward(batch3_in_gradient_unflattened);

        double[][][] leakyrelu2_in_gradient = this.leakyReLU2.backward(tconv2_in_gradient);
        double[] leakyrelu2_in_gradient_flattened = UTIL.flatten(leakyrelu2_in_gradient);
        double[][][] batch2_in_gradient_unflattened = UTIL.unflatten(
                this.batch2.backward(leakyrelu2_in_gradient_flattened),
                leakyrelu2_in_gradient.length, leakyrelu2_in_gradient[0].length, leakyrelu2_in_gradient[0][0].length);


        double[][][] tconv1_in_gradient = this.tconv1.backward(batch2_in_gradient_unflattened);


        double[][][] leakyrelu_in_gradient = this.leakyReLU1.backward(tconv1_in_gradient);
        double[] leakyrelu_in_gradient_flattened = UTIL.flatten(leakyrelu_in_gradient);
        double[] batch1_in_gradient = this.batch1.backward(leakyrelu_in_gradient_flattened);
//        gradient3 = UTIL.multiplyScalar(gradient3, 0.000001);

        double[] dense_in_gradient = this.dense.backward(batch1_in_gradient);

        this.tconv1.updateParameters(batch2_in_gradient_unflattened, learning_rate_gen);
        this.tconv2.updateParameters(batch3_in_gradient_unflattened, learning_rate_gen);
        this.batch1.updateParameters(leakyrelu_in_gradient_flattened, learning_rate_gen);
        this.batch2.updateParameters(leakyrelu2_in_gradient_flattened, learning_rate_gen);
        this.batch3.updateParameters(leakyrelu3_in_gradient_flattened, learning_rate_gen);
        this.dense.updateParameters(batch1_in_gradient, learning_rate_gen);

        if (verbose) {
            // print out the sum of each gradient by flattening it and summing it up using stream().sum()
            System.out.println("Sum of each gradient in generator: ");

            System.out.println("gradient0: " + Arrays.stream(UTIL.flatten(tanh_in_gradient)).sum());
            System.out.println("gradient0_1: " + Arrays.stream(UTIL.flatten(leakyrelu3_in_gradient)).sum());
            System.out.println("gradient0_2: " + Arrays.stream(UTIL.flatten(batch3_in_gradient_unflattened)).sum());
            System.out.println("gradient1: " + Arrays.stream(UTIL.flatten(tconv2_in_gradient)).sum());
            System.out.println("gradient1_2: " + Arrays.stream(UTIL.flatten(leakyrelu2_in_gradient)).sum());
            System.out.println("gradient1_3: " + Arrays.stream(UTIL.flatten(batch2_in_gradient_unflattened)).sum());
            System.out.println("gradient2: " + Arrays.stream(UTIL.flatten(tconv1_in_gradient)).sum());
            System.out.println("gradient2_2: " + Arrays.stream(UTIL.flatten(leakyrelu_in_gradient)).sum());
            System.out.println("out: " + Arrays.stream(leakyrelu_in_gradient_flattened).sum());
            System.out.println("gradient3: " + Arrays.stream(batch1_in_gradient).sum());
            System.out.println("gradient4: " + Arrays.stream(dense_in_gradient).sum());
        }
    }


    public static void main(String[] args) {
        Generator_Implementation generator = new Generator_Implementation();
        double[][][] output = generator.generateImage();

        UTIL.saveImage(UTIL.getBufferedImage(output), "starting_image.png");

        double[][][] outputGradient = new double[output.length][output[0].length][output[0][0].length];

        // loading the first handwritten three from the mnist dataset
        BufferedImage img = UTIL.mnist_load_index(3, 0);

        double[][][] targetOutput = new double[][][]{UTIL.zeroToOneToMinusOneToOne(UTIL.img_to_mat(img))};

        double prev_loss = Double.MAX_VALUE, loss;
        double default_rate = +0.1;
        double smallest = 0.00000000001, largest = 0.1;
        generator.verbose = false;

        for (int epoch = 0, max_epochs = 20000; epoch < max_epochs; epoch++, prev_loss = loss) {
            output = generator.generateImage();

            loss = UTIL.lossMSE(output[0], targetOutput[0]);

            System.out.println(loss);

            UTIL.calculateGradientMSE(outputGradient[0], output[0], targetOutput[0]);

            generator.updateParameters(outputGradient, default_rate);

            if(epoch%10 == 0)
                UTIL.saveImage(UTIL.getBufferedImage(output), "current_image.png");

            if (loss < 0.1) break;
        }

        output = generator.generateImage();
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