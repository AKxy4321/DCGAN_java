package DCGAN.util;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

public class MiscUtils {

    public static final double epsilon = 1e-5;


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

    public static void main(String[] args) {
        double[][][] input = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
        int dz = 1, hz = 1, wz = 1;
        MiscUtils.prettyprint(input);

//        UTIL.prettyprint(UTIL.addZeroesInBetween(input, dz, hz, wz));

        double[] flattened = flatten(input);
        System.out.println(Arrays.toString(flattened));
        double[][][] unflattened = unflatten(flattened, 2, 2, 2);
        MiscUtils.prettyprint(unflattened);

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
}