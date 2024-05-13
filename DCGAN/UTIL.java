package DCGAN;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

public class UTIL {

    public static double epsilon = 1e-5;


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
                outputGradient[i][j] = (1 / (num_values * sqrt_mse)) * (output[i][j] - targetOutput[i][j]);
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

    public static void prettyprint(double[][][] arr) {
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.out.println(Arrays.toString(arr[i][j]));
            }
            System.out.println();
        }
    }

    public static void prettyprint(double[][] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.println(Arrays.toString(arr[i]));
        }
    }


    public static double lossBinaryCrossEntropy(double[] outputs, double[] labels) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += labels[i] * Math.log(outputs[i] + epsilon) + (1 - labels[i]) * Math.log(1 - outputs[i] + epsilon);
        }
        return -loss / outputs.length;
    }

    public static double lossBinaryCrossEntropy(double[][] outputs, double[][] labels) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            for (int j = 0; j < outputs[i].length; j++)
                loss += lossBinaryCrossEntropy(outputs[i][j], labels[i][j]);
        }
        return -loss / (outputs.length * outputs[0].length);
    }

    public static double lossBinaryCrossEntropy(double output, double label) {
        double loss = -(label * Math.log(output + epsilon) - (1 - label) * Math.log(1 - output + epsilon));
        System.out.println(loss + " " + output + " " + label);
        return loss;
    }

    public static double[] gradientBinaryCrossEntropy(double[] outputs, double[] labels) {
        double[] gradient = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            gradient[i] = (outputs[i] - labels[i]) / (outputs[i] * (1 - outputs[i]) + epsilon);
        }
        return gradient;
    }

    public static double gradientBinaryCrossEntropy(double output, double label) {
        return (output - label) / (output * (1 - output) + epsilon);
    }

    public static void calculateBinaryCrossEntropyGradient2D(double[][] outputGradient, double[][] output,
                                                             double[][] targetOutput) {
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                outputGradient[i][j] = gradientBinaryCrossEntropy(output[i][j], targetOutput[i][j]);
            }
        }
    }

    public static BufferedImage load_image(String src) throws IOException {
        BufferedImage file = ImageIO.read(new File(src));
        if (file == null) {
            System.err.println(src);
            throw new IOException("Error loading image");
        }
        return file;
    }

    public static double[][][] unflatten(double[] input, int numFilters, int outputHeight, int outputWidth) {
        double[][][] output = new double[numFilters][outputHeight][outputWidth];
        int index = 0;
        for (int k = 0; k < numFilters; k++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    output[k][h][w] = input[index++];
                }
            }
        }
        return output;
    }

    public static double[][] img_to_mat(BufferedImage imageToPixelate) {
        int w = imageToPixelate.getWidth(), h = imageToPixelate.getHeight();
        int[] pixels = imageToPixelate.getRGB(0, 0, w, h, null, 0, w);
        double[][] dta = new double[w][h];

        for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel++) {
            dta[row][col] = ((pixels[pixel] >> 16 & 0xff)) / 255.0f;
            col++;
            if (col == w) {
                col = 0;
                row++;
            }
        }
        return dta;
    }

    public static BufferedImage mnist_load_index(int label, int index) {
        String mnistPath = Paths.get("DCGAN", "data", "mnist_png", "mnist_png").toString();
        File dir = new File(mnistPath, String.valueOf(label));
        String[] files = dir.list();
        assert files != null;
        String finalPath = mnistPath + File.separator + label + File.separator + files[index];
        try {
            return load_image(finalPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static double[][] mean_1st_layer(double[][][] array) {
        //computes the average of the first layer
        double[][] sum = new double[array[0].length][array[0][0].length];
        for (double[][] doubles : array) {
            for (int j = 0; j < doubles.length; j++) {
                for (int k = 0; k < doubles[j].length; k++) {
                    sum[j][k] += doubles[j][k];
                }
            }
        }
        for (int i = 0; i < sum.length; i++) {
            for (int j = 0; j < sum[i].length; j++) {
                sum[i][j] /= array.length;
            }
        }
        return sum;
    }


    public static double[][] zeroToOneToMinusOneToOne(double[][] array) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                array[i][j] = (array[i][j] - 0.5) * 2.0;
            }
        }
        return array;
    }

    public static double clamp(double val, double lower, double upper) {
        return Math.max(lower, Math.min(val, upper));
    }

    public static double[] clamp(double[] vals, double lower, double upper) {
        double[] clamped = new double[vals.length];
        for (int i = 0; i < vals.length; i++) {
            clamped[i] = clamp(vals[i], lower, upper);
        }
        return clamped;
    }

    public static double[][] multiplyScalar(double[][] array, double scalar) {
        double[][] new_array = new double[array.length][array[0].length];
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                new_array[i][j] = array[i][j] * scalar;
            }
        }
        return new_array;
    }

    public static double[] multiplyScalar(double[] array, double scalar) {
        double[] new_array = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            new_array[i] = array[i] * scalar;
        }
        return new_array;
    }

    public static double gen_loss(double[] fake_output) {
        double[] fake_one = new double[fake_output.length];

        for (int i = 0; i < fake_output.length; i++) {
            fake_one[i] = 1;
        }
        return binary_cross_entropy(fake_one, fake_output);
    }

    public static double disc_loss(double[] real_output, double[] fake_output) {
        double[] real_one = new double[real_output.length];
        double[] fake_zero = new double[fake_output.length];

        for (int i = 0; i < real_output.length; i++) {
            real_one[i] = 1;
            fake_zero[i] = 0;
        }

        double real_loss = binary_cross_entropy(real_one, real_output);
        double fake_loss = binary_cross_entropy(fake_zero, fake_output);

        return real_loss + fake_loss;
    }

    public static double binary_cross_entropy(double[] y_true, double[] y_pred) {
        double sum = 0.0F;
        double epsilon = 1e-5F;
        for (int i = 0; i < y_true.length; i++) {
            sum += y_true[i] * Math.log(y_pred[i]);
        }
        return -sum / y_true.length;
    }

    public static BufferedImage getBufferedImage(double[][] imageData) {
        int width = imageData[0].length;
        int height = imageData.length;

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = imageData[y][x];
                double normalizedValue = (value + 1) / 2.0; // (value - min) / (max - min);
                double brightness = normalizedValue * 255.0;
                int grayValue = (int) brightness;
                int rgb = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    public static BufferedImage getBufferedImage(double[][][] imageData) {
        // TODO: change this to get the Buffered Image taking into accoun that this might have RGB channels in the extra 3rd dimension
        return getBufferedImage(imageData[0]);
    }

    public static double[][] unflatten(double[] out_l, int height, int width) {
        double[][] output = new double[height][width];
        int k = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[i][j] = out_l[k++];
            }
        }
        return output;
    }


    public static double[] flatten(double[][] input) {
        int totalLength = 0;
        for (double[] arr : input) {
            totalLength += arr.length;
        }
        double[] output = new double[totalLength];
        int k = 0;
        for (double[] doubles : input) {
            for (double adouble : doubles) {
                output[k++] = adouble;
            }
        }
        return output;
    }

    public static double[] flatten(double[][][] input) {
        int actualDepth = input.length;
        int actualHeight = input[0].length;
        int actualWidth = input[0][0].length;
        int m = 0;
        double[] flatten_output = new double[actualDepth * actualHeight * actualWidth];
        for (double[][] doubles : input) {
            for (int i = 0; i < actualHeight; i++) {
                for (int j = 0; j < actualWidth; j++) {
                    flatten_output[m++] = doubles[i][j];
                }
            }
        }
        return flatten_output;
    }

    public static void main(String[] args) {
        double[][][] input = {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}}
        };
        int dz = 1, hz = 1, wz = 1;
        UTIL.prettyprint(input);

//        UTIL.prettyprint(UTIL.addZeroesInBetween(input, dz, hz, wz));

        double[] flattened = flatten(input);
        System.out.println(Arrays.toString(flattened));
        double[][][] unflattened = unflatten(flattened, 2, 2, 2);
        UTIL.prettyprint(unflattened);

//        UTIL.prettyprint(UTIL.rotate180(new double[][]{
//                {1, 2, 3, 1},
//                {4, 5, 6, 1},
//                {7, 8, 9, 1},
//                {2, 2, 2, 2}}));

//        double[][][] array = new double[][][]{
//                {
//                        {1, 2, 3},
//                        {4, 5, 6},
//                        {7, 8, 9}
//                },
//                {
//                        {10, 11, 12},
//                        {13, 14, 15},
//                        {16, 17, 18}
//                }
//        };
//        double[] flatten = flatten(array);
//        double[][][] unflatten = unflatten(flatten, 2, 3, 3);
//
//        System.out.println("Actual array : ");
//        for (double[][] doubles : array) {
//            for (double[] aDouble : doubles) {
//                for (double v : aDouble) {
//                    System.out.print(v + " ");
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }
//
//        System.out.println("Flattened array : ");
//        System.out.println(java.util.Arrays.toString(flatten));
//
//        System.out.println("Unflattened version of flattened array : ");
//        for (double[][] doubles : unflatten) {
//            for (double[] aDouble : doubles) {
//                for (double v : aDouble) {
//                    System.out.print(v + " ");
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }

    }

    public static double[] negate(double[] array) {
        double[] new_array = new double[array.length];
        for (int i = 0; i < array.length; i++)
            new_array[i] = array[i] * -1;
        return new_array;
    }

    public static double mean(double[] array) {
        double sum = 0;
        for (double genLoss : array) {
            sum += genLoss;
        }
        return sum / array.length;
    }

    public static double[] mean_1st_layer(double[][] array) {
        double[] sum = new double[array[0].length];
        for (double[] subarray : array) {
            for (int j = 0; j < subarray.length; j++) {
                sum[j] += subarray[j];
            }
        }
        for (int i = 0; i < sum.length; i++) {
            sum[i] /= array.length;
        }
        return sum;
    }

    public static double[][][] mean_1st_layer(double[][][][] array) {
        double[][][] sum = new double[array[0].length][array[0][0].length][array[0][0][0].length];
        for (double[][][] subarray : array) {
            for (int i = 0; i < subarray.length; i++) {
                for (int j = 0; j < subarray[0].length; j++) {
                    for (int k = 0; k < subarray[0][0].length; k++) {
                        sum[i][j][k] += subarray[i][j][k];
                    }
                }
            }
        }
        for (int i = 0; i < sum.length; i++) {
            for (int j = 0; j < sum[0].length; j++) {
                for (int k = 0; k < sum[0][0].length; k++) {
                    sum[i][j][k] /= array.length;
                }
            }
        }
        return sum;
    }


    public static void saveImage(BufferedImage image, String name) {
        File outputImageFile = new File(name);
        try {
            ImageIO.write(image, "png", outputImageFile);
//                    System.out.println("Image saved successfully to: " + outputImageFile.getAbsolutePath());
            System.out.println("Saving image");
        } catch (Exception e) {
            System.err.println("Error saving image: " + e.getMessage());
        }
    }

    public static double[][][][] multiplyScalar(double[][][][] array, double scalar) {
        double[][][][] new_array = new double[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        new_array[i][j][k][l] = array[i][j][k][l] * scalar;
        return new_array;
    }


    public static double sum(double[][] array) {
        double sum = 0;
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                sum += array[i][j];
        return sum;
    }


    public static double[][] rotate180(double[][] array) {
        double[][] rotated = new double[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                rotated[i][j] = array[array.length - i - 1][array[0].length - j - 1];
        return rotated;
    }

    public static double[][] rotate90(double[][] array) {
        double[][] rotated = new double[array[0].length][array.length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                rotated[j][array.length - i - 1] = array[i][j];
        return rotated;
    }

    public static double[][][] rotate180(double[][][] array) {
        double[][][] rotated = new double[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    rotated[i][j][k] = array[i][array[0].length - j - 1][array[0][0].length - k - 1];
        return rotated;
    }

    public static double[][] transpose(double[][] array) {
        double[][] transposed = new double[array[0].length][array.length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                transposed[j][i] = array[i][j];
        return transposed;
    }

    public static double[][][] transpose(double[][][] array) {
        double[][][] transposed = new double[array.length][array[0][0].length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    transposed[i][k][j] = array[i][j][k];
        return transposed;
    }

    public static void incrementArrayByArray(double[][][] resultArray, double[][][] array) {
        /**
         * Adds the values of array to resultArray. No new array is made for this
         * */
        for (int i = 0; i < resultArray.length; i++)
            for (int j = 0; j < resultArray[0].length; j++)
                for (int k = 0; k < resultArray[0][0].length; k++)
                    resultArray[i][j][k] += array[i][j][k];
    }


    public static double[][][] multiplyScalar(double[][][] array, double scalar) {
        double[][][] new_array = new double[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    new_array[i][j][k] = array[i][j][k] * scalar;
        return new_array;
    }

    public static double[][][] gradientRMSE(double[][][] output, double[][][] targetOutput) {
        double[][][] gradientArray = new double[output.length][output[0].length][output[0][0].length];
        calculateGradientRMSE(gradientArray, output, targetOutput);
        return gradientArray;
    }

    public static double[][][] addZeroesInBetween(double[][][] input, int dz, int hz, int wz) {

        double[][][] output = new double
                [input.length + (input.length - 1) * (dz)]
                [input[0].length + (input[0].length - 1) * (hz)]
                [input[0][0].length + (input[0][0].length - 1) * (wz)];

//        UTIL.prettyprint(output);

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                for (int k = 0; k < input[0][0].length; k++) {

                    output[i + i * dz][j + j * hz][k + k * wz] = input[i][j][k];

                }
            }
        }

        return output;

    }
}