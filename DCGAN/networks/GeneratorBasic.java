package DCGAN.networks;

import DCGAN.util.MiscUtils;
import DCGAN.util.XavierInitializer;
import DCGAN.layers.*;

import java.util.Arrays;

import static DCGAN.util.MiscUtils.*;
import static DCGAN.util.TrainingUtils.gradientRMSE;
import static DCGAN.util.TrainingUtils.lossRMSE;

public class GeneratorBasic {
    DenseLayer dense1 = new DenseLayer(100, 256);
    SigmoidLayer sigmoid1 = new SigmoidLayer();
    DenseLayer dense2 = new DenseLayer(dense1.outputSize, 7*7*29);
    TransposeConvolutionalLayer transposeConv1 = new TransposeConvolutionalLayer(3, 17, 1, 7, 7, 29, 0, false);
    // output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding
    TransposeConvolutionalLayer tconv2 = new TransposeConvolutionalLayer(1, 1, 1, transposeConv1.outputWidth, transposeConv1.outputHeight, transposeConv1.outputDepth, 0, false);
    TanhLayer tanh = new TanhLayer();

    public double[][][] forward(double[] input) {
        double[] dense1Out = dense1.forward(input);
        double[] sigmoid1Out = sigmoid1.forward(dense1Out);
        double[] dense2Out = dense2.forward(sigmoid1Out);
        double[][][] dense2OutUnflattened = MiscUtils.unflatten(dense2Out, transposeConv1.inputDepth, transposeConv1.inputHeight, transposeConv1.inputWidth);
        double[][][] transposeConv1Out = transposeConv1.forward(dense2OutUnflattened);
        System.out.println("transposeConv1Out Shape : " + transposeConv1Out.length + " " + transposeConv1Out[0].length + " " + transposeConv1Out[0][0].length);
        double[][][] tconv2Out = tconv2.forward(transposeConv1Out);
        double[][][] tanhOut = tanh.forward(tconv2Out);
        return tanhOut;
    }

    public void updateParameters(double[][][] gradOutput, double learningRate) {
        double[][][] grad_tanh_inputgrad_tconv2_outputgrad = tanh.backward(gradOutput);
        double[][][] grad_tconv2_inputgrad_tconv1_outputgrad = tconv2.backward(grad_tanh_inputgrad_tconv2_outputgrad);
        double[][][] gradTransposeConv1_inputgrad_dense2_outputgrad = transposeConv1.backward(grad_tconv2_inputgrad_tconv1_outputgrad);
        double[] gradDense2_inputgrad_sigmoid1_outputgrad = dense2.backward(MiscUtils.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad));
        double[] gradSigmoid1_inputgrad_dense1_outputgrad = sigmoid1.backward(gradDense2_inputgrad_sigmoid1_outputgrad);
        double[] gradDense1 = dense1.backward(gradSigmoid1_inputgrad_dense1_outputgrad);

        tconv2.updateParameters(grad_tanh_inputgrad_tconv2_outputgrad, learningRate);
        transposeConv1.updateParameters(grad_tconv2_inputgrad_tconv1_outputgrad, learningRate);
        dense2.updateParameters(MiscUtils.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad), learningRate);
        dense1.updateParameters(gradSigmoid1_inputgrad_dense1_outputgrad, learningRate);

        System.out.println("sum of each gradient: ");
        System.out.println("gradDense1 : " + Arrays.stream(gradDense1).sum());
        System.out.println("gradSigmoid1_inputgrad_dense1_outputgrad : " + Arrays.stream(gradSigmoid1_inputgrad_dense1_outputgrad).sum());
        System.out.println("gradDense2_inputgrad_sigmoid1_outputgrad : " + Arrays.stream(gradDense2_inputgrad_sigmoid1_outputgrad).sum());
        System.out.println("gradTransposeConv1_inputgrad_dense2_outputgrad : " + Arrays.stream(MiscUtils.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad)).sum());
    }

    public static void main(String[] args) {

        GeneratorBasic generator = new GeneratorBasic();
        double[] input = new double[100];
        Arrays.fill(input, 1);
        double[][][] targetOutput = new double[generator.tconv2.outputDepth][generator.tconv2.outputHeight][generator.tconv2.outputWidth];
        for (int i = 0; i < targetOutput.length; i++)
            for (int j = 0; j < targetOutput[0].length; j++)
                Arrays.fill(targetOutput[i][j], 1);
        targetOutput = new double[][][]{
                {
                        {0, 0, 0, 1, 1, 1, 0, 0, 0},
                        {0, 0, 1, 1, 1, 1, 1, 1, 0},
                        {0, 1, 1, 0, 0, 0, 1, 1, 0},
                        {0, 1, 1, 0, 0, 0, 1, 1, 0},
                        {0, 1, 1, 0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 1, 1, 1, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 1, 1, 1, 0},
                        {0, 0, 0, 0, 1, 1, 1, 1, 0},
                }
        };
        saveImage(getBufferedImage(targetOutput[0]), "targetOutput.png");
//        // replace all 0s with -1s
//        for (int i = 0; i < targetOutput.length; i++)
//            for (int j = 0; j < targetOutput[0].length; j++)
//                for (int k = 0; k < targetOutput[0][0].length; k++)
//                    if (targetOutput[i][j][k] == 0)
//                        targetOutput[i][j][k] = -1;

//        System.out.println("Target output :");
        prettyprint(targetOutput);

        System.out.println(targetOutput.length + " " + targetOutput[0].length + " " + targetOutput[0][0].length);
        for (int epoch = 0; epoch < 100000; epoch++) {
            double[][][] output = generator.forward(XavierInitializer.xavierInit1D(generator.dense1.inputSize));
            System.out.println("output shape : " + output.length + " " + output[0].length + " " + output[0][0].length);
            System.out.println("targetOutput shape : " + targetOutput.length + " " + targetOutput[0].length + " " + targetOutput[0][0].length);
            double[][][] gradOutput = gradientRMSE(output, targetOutput);
            double loss = lossRMSE(output, targetOutput);
            System.err.println("loss : " + loss);
            generator.updateParameters(gradOutput, 0.001);
            if (epoch % 100 == 0) {
//                UTIL.prettyprint(output);
                saveImage(getBufferedImage(generator.forward(input)[0]), "actual_output.png");
            }
        }
    }

    public double[][][] generateImage() {
        return forward(XavierInitializer.xavierInit1D(dense1.inputSize));
    }
}