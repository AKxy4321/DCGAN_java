package DCGAN.networks;

import DCGAN.util.MiscUtils;
import DCGAN.util.XavierInitializer;
import DCGAN.layers.*;

import java.util.Arrays;

import static DCGAN.util.TrainingUtils.gradientRMSE;
import static DCGAN.util.TrainingUtils.lossRMSE;

public class GeneratorBasicStrideGreaterThanOne {
    DenseLayer dense1 = new DenseLayer(100, 128);
    SigmoidLayer sigmoid1 = new SigmoidLayer();
    DenseLayer dense2 = new DenseLayer(dense1.outputSize, 7 * 7 * 47);
    TransposeConvolutionalLayer transposeConv1 = new TransposeConvolutionalLayer(3, 43, 2, 7, 7, 47, 0, false);
    // this.stride * (inputHeight - 1) + filterSize - 2 * padding; = 2 * (7 - 1) + 3 - 2 * 0 = 15
    LeakyReLULayer leakyReLU = new LeakyReLULayer();
    TransposeConvolutionalLayer tconv2 = new TransposeConvolutionalLayer(3, 1, 1, transposeConv1.outputWidth, transposeConv1.outputHeight, transposeConv1.outputDepth, 0, false);
    // (15-1) * 1 + 3 - 2 * 0 = 17
    TanhLayer tanh = new TanhLayer();

    public double[][][] forward(double[] input) {
        double[] dense1Out = dense1.forward(input);
        double[] sigmoid1Out = sigmoid1.forward(dense1Out);
        double[] dense2Out = dense2.forward(sigmoid1Out);
        double[][][] dense2OutUnflattened = MiscUtils.unflatten(dense2Out, transposeConv1.inputDepth, transposeConv1.inputHeight, transposeConv1.inputWidth);
        double[][][] transposeConv1Out = transposeConv1.forward(dense2OutUnflattened);
        System.out.println("transposeConv1Out Shape : " + transposeConv1Out.length + " " + transposeConv1Out[0].length + " " + transposeConv1Out[0][0].length);
        double[][][] leakyReLUOut = leakyReLU.forward(transposeConv1Out);
        double[][][] tconv2Out = tconv2.forward(leakyReLUOut);
        double[][][] tanhOut = tanh.forward(tconv2Out);
        return tanhOut;
    }

    public void updateParameters(double[][][] gradOutput, double learningRate) {
        double[][][] grad_tanh_inputgrad_tconv2_outputgrad = tanh.backward(gradOutput);
        double[][][] grad_tconv2_inputgrad_tconv1_outputgrad = tconv2.backward(grad_tanh_inputgrad_tconv2_outputgrad);
        double[][][] grad_leakyReLU_inputgrad_tconv1_outputgrad = leakyReLU.backward(grad_tconv2_inputgrad_tconv1_outputgrad);
        double[][][] gradTransposeConv1_inputgrad_dense2_outputgrad = transposeConv1.backward(grad_leakyReLU_inputgrad_tconv1_outputgrad);
        double[] gradDense2_inputgrad_sigmoid1_outputgrad = dense2.backward(MiscUtils.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad));
        double[] gradSigmoid1_inputgrad_dense1_outputgrad = sigmoid1.backward(gradDense2_inputgrad_sigmoid1_outputgrad);
        double[] gradDense1 = dense1.backward(gradSigmoid1_inputgrad_dense1_outputgrad);

        tconv2.updateParameters(grad_tanh_inputgrad_tconv2_outputgrad);
        transposeConv1.updateParameters(grad_leakyReLU_inputgrad_tconv1_outputgrad);
        dense2.updateParameters(MiscUtils.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad));
        dense1.updateParameters(gradSigmoid1_inputgrad_dense1_outputgrad);

        System.out.println("sum of each gradient: ");
        System.out.println("gradDense1 : " + Arrays.stream(gradDense1).sum());
        System.out.println("gradSigmoid1_inputgrad_dense1_outputgrad : " + Arrays.stream(gradSigmoid1_inputgrad_dense1_outputgrad).sum());
        System.out.println("gradDense2_inputgrad_sigmoid1_outputgrad : " + Arrays.stream(gradDense2_inputgrad_sigmoid1_outputgrad).sum());
        System.out.println("gradTransposeConv1_inputgrad_dense2_outputgrad : " + Arrays.stream(MiscUtils.flatten(gradTransposeConv1_inputgrad_dense2_outputgrad)).sum());
    }

    public static void main(String[] args) {

        GeneratorBasicStrideGreaterThanOne generator = new GeneratorBasicStrideGreaterThanOne();
        double[] input = new double[100];
        Arrays.fill(input, 1);
        double[][][] targetOutput = new double[generator.tconv2.outputDepth][generator.tconv2.outputHeight][generator.tconv2.outputWidth];
        for (int i = 0; i < targetOutput.length; i++)
            for (int j = 0; j < targetOutput[0].length; j++)
                Arrays.fill(targetOutput[i][j], 1);

//        System.out.println("Target output :");
        MiscUtils.prettyprint(targetOutput);
        MiscUtils.saveImage(MiscUtils.getBufferedImage(targetOutput[0]), "targetOutput.png");

        System.out.println("targetOutput Shape : " + targetOutput.length + " " + targetOutput[0].length + " " + targetOutput[0][0].length);

        System.out.println(targetOutput.length + " " + targetOutput[0].length + " " + targetOutput[0][0].length);
        for (int epoch = 0; epoch < 100000; epoch++) {
            double[][][] output = generator.forward(XavierInitializer.xavierInit1D(generator.dense1.inputSize));

            System.out.println("output Shape : " + output.length + " " + output[0].length + " " + output[0][0].length);
            System.out.println("output shape according to formula : " + generator.tconv2.outputDepth + "x" + generator.tconv2.outputHeight + "x" + generator.tconv2.outputWidth);

            double[][][] gradOutput = gradientRMSE(output, targetOutput);
            double loss = lossRMSE(output, targetOutput);
            System.err.println("loss : " + loss);
            generator.updateParameters(gradOutput, 0.1);
            if (epoch % 100 == 0) {
//                UTIL.prettyprint(output);
                MiscUtils.saveImage(MiscUtils.getBufferedImage(generator.forward(input)[0]), "actual_output_non_one_stride.png");
            }
        }

    }

    public double[][][] generateImage() {
        return forward(XavierInitializer.xavierInit1D(dense1.inputSize));
    }
}