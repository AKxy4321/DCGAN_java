package DCGAN;

import java.util.Arrays;

public class GeneratorBasic {
    DenseLayer dense1 = new DenseLayer(100, 128);
    SigmoidLayer sigmoid1 = new SigmoidLayer();
    DenseLayer dense2 = new DenseLayer(dense1.outputSize, 49*7);
    TransposeConvolutionalLayer transposeConv1 = new TransposeConvolutionalLayer(3, 3, 1, 7, 7, 7, 0, false);

    TransposeConvolutionalLayer tconv2 = new TransposeConvolutionalLayer(1, 1, 1, transposeConv1.outputWidth, transposeConv1.outputHeight, transposeConv1.outputDepth, 0, false);

    public double[][][] forward(double[] input) {
        double[] dense1Out = dense1.forward(input);
        double[] sigmoid1Out = sigmoid1.forward(dense1Out);
        double[] dense2Out = dense2.forward(sigmoid1Out);
        double[][][] dense2OutUnflattened = UTIL.unflatten(dense2Out, transposeConv1.inputDepth, transposeConv1.inputHeight, transposeConv1.inputWidth);
        double[][][] transposeConv1Out = transposeConv1.forward(dense2OutUnflattened);
        System.out.println("transposeConv1Out Shape : " + transposeConv1Out.length + " " + transposeConv1Out[0].length + " " + transposeConv1Out[0][0].length);
        double[][][] tconv2Out = tconv2.forward(transposeConv1Out);
        return tconv2Out;
    }

    public void updateParameters(double[][][] gradOutput, double learningRate) {
        double[][][] grad_tconv2_inputgrad_tconv1_outputgrad = tconv2.backward(gradOutput);
        double[][][] gradTransposeConv1_inputgrad_dense2_outputgrad = transposeConv1.backward(grad_tconv2_inputgrad_tconv1_outputgrad);
        double[] gradDense2_inputgrad_sigmoid1_outputgrad = dense2.backward(UTIL.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad));
        double[] gradSigmoid1_inputgrad_dense1_outputgrad = sigmoid1.backward(gradDense2_inputgrad_sigmoid1_outputgrad);
        double[] gradDense1 = dense1.backward(gradSigmoid1_inputgrad_dense1_outputgrad);

        tconv2.updateParameters(gradOutput, learningRate);
        transposeConv1.updateParameters(grad_tconv2_inputgrad_tconv1_outputgrad, learningRate);
        dense2.updateParameters(UTIL.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad), learningRate);
        dense1.updateParameters(gradSigmoid1_inputgrad_dense1_outputgrad, learningRate);

        System.out.println("sum of each gradient: ");
        System.out.println("gradDense1 : " + Arrays.stream(gradDense1).sum());
        System.out.println("gradSigmoid1_inputgrad_dense1_outputgrad : " + Arrays.stream(gradSigmoid1_inputgrad_dense1_outputgrad).sum());
        System.out.println("gradDense2_inputgrad_sigmoid1_outputgrad : " + Arrays.stream(gradDense2_inputgrad_sigmoid1_outputgrad).sum());
        System.out.println("gradTransposeConv1_inputgrad_dense2_outputgrad : " + Arrays.stream(UTIL.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad)).sum());
    }

    public static void main(String[] args) {
        GeneratorBasic generator = new GeneratorBasic();
        double[] input = new double[100];
        Arrays.fill(input, 1);
        double[][][] targetOutput = new double[generator.tconv2.outputDepth][generator.tconv2.outputHeight][generator.tconv2.outputWidth];

        System.out.println("Target output :");
        UTIL.prettyprint(targetOutput);

        for (int epoch = 0; epoch < 1000; epoch++) {
            double[][][] output = generator.forward(input);
            double[][][] gradOutput = UTIL.gradientMSE(output, targetOutput);
            double loss = UTIL.lossMSE(output, targetOutput);
            System.err.println("loss : " + loss);
            generator.updateParameters(gradOutput, 0.0001);
        }

    }
}