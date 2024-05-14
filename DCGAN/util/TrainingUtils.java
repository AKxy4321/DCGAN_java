package DCGAN.util;

public class TrainingUtils {
    public static final double epsilon = 1e-6;

    public static double lossBinaryCrossEntropy(double[] outputs, double[] labels) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += labels[i] * Math.log(outputs[i] + epsilon) + (1 - labels[i]) * Math.log(1 - outputs[i] + epsilon);
        }
        return -loss / outputs.length;
    }

    public static double[] gradientBinaryCrossEntropy(double[] outputs, double[] labels) {
        double[] gradient = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            gradient[i] = (outputs[i] - labels[i]) / (outputs[i] * (1 - outputs[i]) + epsilon);
        }
        return gradient;
    }

    public static double lossMSE(double[] outputs, double[] expectedOutputs) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += Math.pow(outputs[i] - expectedOutputs[i], 2);
        }
        return (loss / outputs.length);
    }

    public static double lossMSE(double[][] outputs, double[][] expectedOutputs) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            for (int j = 0; j < outputs[0].length; j++)
                loss += Math.pow(outputs[i][j] - expectedOutputs[i][j], 2);
        }
        return (loss / (outputs.length * outputs[0].length));
    }

    public static double lossMSE(double[][][] output, double[][][] targetOutput) {
        double loss = 0;
        for (int i = 0; i < output.length; i++)
            for (int j = 0; j < output[0].length; j++)
                for (int k = 0; k < output[0][0].length; k++)
                    loss += Math.pow(output[i][j][k] - targetOutput[i][j][k], 2);
        return loss / (output.length * output[0].length * output[0][0].length);
    }

    public static double gradientSquaredError(double output, double expectedOutput) {
        return 2 * (output - expectedOutput);
    }

    public static void calculateGradientMSE(double[][] outputGradient, double[][] output, double[][] targetOutput) {
        double num_values = output.length * output[0].length;
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                outputGradient[i][j] = gradientSquaredError(output[i][j], targetOutput[i][j]) / num_values;
            }
        }
    }

    public static double[] gradientMSE(double[] outputs, double[] expectedOutputs) {
        double[] gradients = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            gradients[i] = 2 * (outputs[i] - expectedOutputs[i]) / outputs.length;
        }
        return gradients;
    }

    public static double[][][] gradientMSE(double[][][] outputs, double[][][] expectedOutputs) {
        double[][][] gradients = new double[outputs.length][outputs[0].length][outputs[0][0].length];
        for (int i = 0; i < outputs.length; i++) {
            for (int j = 0; j < outputs[0].length; j++) {
                for (int k = 0; k < outputs[0][0].length; k++) {
                    gradients[i][j][k] = 2 * (outputs[i][j][k] - expectedOutputs[i][j][k]) / (outputs.length * outputs[0].length * outputs[0][0].length);
                }
            }
        }
        return gradients;
    }


    public static double lossRMSE(double[][] outputs, double[][] expectedOutputs) {
        double mse = lossMSE(outputs, expectedOutputs);
        return Math.sqrt(mse);
    }

    public static void calculateGradientRMSE(double[][] outputGradient, double[][] output, double[][] targetOutput) {
        double num_values = output.length * output[0].length;
        double sqrt_mse = Math.sqrt(lossMSE(output, targetOutput));
        for (int i = 0; i < outputGradient.length; i++) {
            for (int j = 0; j < outputGradient[0].length; j++) {
                outputGradient[i][j] = (1 / (num_values * sqrt_mse + epsilon)) * (output[i][j] - targetOutput[i][j]);
            }
        }
    }


    public static double lossRMSE(double[][][] output, double[][][] targetOutput) {
        double loss = 0;
        for (int i = 0; i < output.length; i++)
            for (int j = 0; j < output[0].length; j++)
                for (int k = 0; k < output[0][0].length; k++)
                    loss += Math.pow(output[i][j][k] - targetOutput[i][j][k], 2);
        return Math.sqrt(loss / (output.length * output[0].length * output[0][0].length));
    }

    public static void calculateGradientRMSE(double[][][] outputGradient, double[][][] output, double[][][] targetOutput) {
        double num_values = output.length * output[0].length * output[0][0].length;
        double sqrt_mse = Math.sqrt(lossMSE(output, targetOutput));
        for (int i = 0; i < outputGradient.length; i++) {
            for (int j = 0; j < outputGradient[0].length; j++) {
                for (int k = 0; k < outputGradient[0].length; k++) {
                    outputGradient[i][j][k] = (1 / (num_values * sqrt_mse + epsilon)) * (output[i][j][k] - targetOutput[i][j][k]);
                }
            }
        }
    }

    public static double[][][] gradientRMSE(double[][][] output, double[][][] targetOutput) {
        double[][][] gradientArray = new double[output.length][output[0].length][output[0][0].length];
        calculateGradientRMSE(gradientArray, output, targetOutput);
        return gradientArray;
    }

}
