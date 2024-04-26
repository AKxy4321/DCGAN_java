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

    public Generator_Implementation() {
        int noise_length = 100;
        int tconv1_input_width = 7, tconv1_input_height = 7, tconv1_input_depth = 256;
        this.dense_output_size = tconv1_input_width * tconv1_input_height * tconv1_input_depth;
        this.dense = new DenseLayer(noise_length, this.dense_output_size);
        this.batch1 = new BatchNormalization();
        this.leakyReLU1 = new LeakyReLULayer();

//         output_shape = (input_shape - 1) * stride  + kernel_size - 2 * padding
//         padding = ((filter_height - 1) * stride + output_height - input_height)
// if padding = same : ((filter_height - 1) * stride + 0) = (5-1)*1
        this.tconv1 = new TransposeConvolutionalLayer(5, 128, 1, tconv1_input_width, tconv1_input_height, tconv1_input_depth, 0);
        //11*11
        this.batch2 = new BatchNormalization();
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(5, 64, 2, tconv1.outputWidth, tconv1.outputHeight, tconv1.outputDepth, 3);
        this.batch3 = new BatchNormalization();
        this.leakyReLU3 = new LeakyReLULayer();
        this.tconv3 = new TransposeConvolutionalLayer(6, 1, 2, tconv2.outputWidth, tconv2.outputHeight, tconv2.outputDepth, 7);
        this.tanh = new TanhLayer();
        // size of tanh is supposed to be 64x5*5
        /*
         * [128,1,6,6]   [3,3]   2
         * 128,1,13,13   [4,4]   2
         *  128,1,28,28
         * */

    }

    public double[][][] generateImage() {
        double[] noise = XavierInitializer.xavierInit1D(100); // generate noise input that we want to pass to the generator

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

        double[][][] gradient0 = this.tanh.backward(gen_output_gradient);
        double[][][] gradient0_1 = this.leakyReLU3.backward(gradient0);
        double[] gradient_0_1_flattened = UTIL.flatten(gradient0_1);
        double[][][] gradient0_2 = UTIL.unflatten(this.batch3.backward(gradient_0_1_flattened),
                gradient0_1.length, gradient0_1[0].length, gradient0_1[0][0].length);
        this.batch3.updateParameters(gradient_0_1_flattened, learning_rate_gen);

        double[][][] gradient1 = this.tconv2.backward(gradient0_2);
        this.tconv2.updateParameters(gradient0_2, learning_rate_gen);

        double[][][] gradient1_2 = this.leakyReLU2.backward(gradient1);
        double[] gradient1_2_flattened = UTIL.flatten(gradient1_2);
        double[][][] gradient1_3 = UTIL.unflatten(
                this.batch2.backward(gradient1_2_flattened),
                gradient1_2.length, gradient1_2[0].length, gradient1_2[0][0].length);
        this.batch2.updateParameters(gradient1_2_flattened, learning_rate_gen);

        double[][][] gradient2 = this.tconv1.backward(gradient1_3);
        this.tconv1.updateParameters(gradient1_3, learning_rate_gen);

        double[][][] gradient2_2 = this.leakyReLU1.backward(gradient2);
        double[] out = UTIL.flatten(gradient2_2);
        double[] gradient3 = this.batch1.backward(out);
//        gradient3 = UTIL.multiplyScalar(gradient3, 0.000001);
        this.batch1.updateParameters(out, learning_rate_gen * 1); // 0.000001);

        double[] gradient4 = this.dense.backward(gradient3);
        this.dense.updateParameters(gradient3, learning_rate_gen * 1);// 0.0000000001);

        // print out the sum of each gradient by flattening it and summing it up using stream().sum()
        System.out.println("Sum of each gradient in generator: ");

        System.out.println("gradient0: " + Arrays.stream(UTIL.flatten(gradient0)).sum());
        System.out.println("gradient0_1: " + Arrays.stream(UTIL.flatten(gradient0_1)).sum());
        System.out.println("gradient0_2: " + Arrays.stream(UTIL.flatten(gradient0_2)).sum());
        System.out.println("gradient1: " + Arrays.stream(UTIL.flatten(gradient1)).sum());
        System.out.println("gradient1_2: " + Arrays.stream(UTIL.flatten(gradient1_2)).sum());
        System.out.println("gradient1_3: " + Arrays.stream(UTIL.flatten(gradient1_3)).sum());
        System.out.println("gradient2: " + Arrays.stream(UTIL.flatten(gradient2)).sum());
        System.out.println("gradient2_2: " + Arrays.stream(UTIL.flatten(gradient2_2)).sum());
        System.out.println("out: " + Arrays.stream(out).sum());
        System.out.println("gradient3: " + Arrays.stream(gradient3).sum());
        System.out.println("gradient4: " + Arrays.stream(gradient4).sum());
    }


    public static void main(String[] args) {
        Generator_Implementation generator = new Generator_Implementation();
        double[][][] output = generator.generateImage();
        BufferedImage image = UTIL.getBufferedImage(output);

        UTIL.saveImage(image, "generated_image.png");
        System.out.println(output.length + " " + output[0].length + " " + output[0][0].length);

        generator.updateParameters(new double[output.length][output[0].length][output[0][0].length], 0.01);
    }
}