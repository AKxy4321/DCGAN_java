package DCGAN.layers;

import java.io.Serializable;

public class SigmoidLayer implements Serializable {
    private static final long serialVersionUID = 1L;
    public double[] input1D;
    public double[] output1D;

    public double apply_sigmoid(double x) {
        return 1 /( 1 + Math.exp(-x));
    }

    public double[] forward(double[] input) {
        this.input1D = input;
        int size = input.length;
        output1D = new double[size];

        for (int i = 0; i < size; i++) {
            output1D[i] = apply_sigmoid(input[i]);
        }
        return output1D;
    }


    @Deprecated
    public double[] backward(double[] outputGradient) {
        return backward(outputGradient, this.output1D);
    }

    public double[] backward(double[] outputGradient, double[] output) {
        double[] inputGradient = new double[outputGradient.length];
        int size = outputGradient.length;

        for (int i = 0; i < size; i++) {
            inputGradient[i] = (1-output[i])*output[i] * outputGradient[i];
        }
        return inputGradient;
    }
}
