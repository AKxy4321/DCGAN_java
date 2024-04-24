package DCGAN;

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
    TanhLayer tanh;

    public Generator_Implementation() {
        this.dense_output_size = 7 * 7 * 128;
        this.dense = new DenseLayer(100, this.dense_output_size);
        this.batch1 = new BatchNormalization();
        this.leakyReLU1 = new LeakyReLULayer();
        this.tconv1 = new TransposeConvolutionalLayer(256, 5, 64, 1);
        this.batch2 = new BatchNormalization();
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(64, 8, 1, 2);
        this.batch3 = new BatchNormalization();
        this.leakyReLU3 = new LeakyReLULayer();
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
        double[][][] gen_batch1_output_unflattened = UTIL.unflatten(gen_batch1_output, 128, 7, 7);
        double[][][] gen_leakyrelu_output1 = this.leakyReLU1.forward(gen_batch1_output_unflattened);
        double[][][] outputTconv1 = this.tconv1.forward(gen_leakyrelu_output1);
        double[] gen_batch2_output = this.batch2.forward(UTIL.flatten(outputTconv1), true);
        double[][][] gen_batch2_output_unflattened = UTIL.unflatten(gen_batch2_output, outputTconv1.length, outputTconv1[0].length, outputTconv1[0][0].length);
        double[][][] gen_leakyrelu_output2 = this.leakyReLU2.forward(gen_batch2_output_unflattened);
        double[][][] outputTconv2 = this.tconv2.forward(gen_leakyrelu_output2);
        double[] gen_batch3_output = this.batch3.forward(UTIL.flatten(outputTconv2), true);
        double[][][] gen_batch3_output_unflattened = UTIL.unflatten(gen_batch3_output, outputTconv2.length, outputTconv2[0].length, outputTconv2[0][0].length);
        double[][][] gen_leakyrelu_output3 = this.leakyReLU3.forward(gen_batch3_output_unflattened);
        double[][][] fakeImage = this.tanh.forward(gen_leakyrelu_output3);
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
        this.batch1.updateParameters(out, learning_rate_gen);

        double[] gradient4 = this.dense.backward(gradient3);
        this.dense.updateParameters(gradient3, learning_rate_gen);
    }
}