package DCGAN;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class UTIL {
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

    public static double[][] multiplyScalar(double[][] array, double scalar) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                array[i][j] *= scalar;
            }
        }
        return array;
    }

    public double gen_loss(double[] fake_output) {
        double[] fake_one = new double[fake_output.length];

        for (int i = 0; i < fake_output.length; i++) {
            fake_one[i] = 1;
        }
        return binary_cross_entropy(fake_one, fake_output);
    }

    public double disc_loss(double[] real_output, double[] fake_output) {
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

    public double binary_cross_entropy(double[] y_true, double[] y_pred) {
        double sum = 0.0F;
        double epsilon = 1e-5F;
        for (int i = 0; i < y_true.length; i++) {
            sum += y_true[i] * Math.log(y_pred[i]+epsilon) + (1 - y_true[i]) * Math.log(1 - y_pred[i]+epsilon);
        }
        return -sum / y_true.length;
    }

    public static BufferedImage getBufferedImage(double[][][] fakeImage) {
        double max = fakeImage[0][0][0];
        double min = fakeImage[0][0][0];
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
//        for (int y = 0; y < 28; y++) {
//            for (int x = 0; x < 28; x++) {
//                if (max < fakeImage[0][y][x]) {
//                    max = fakeImage[0][y][x];
//                }
//                if (min > fakeImage[0][y][x]) {
//                    min = fakeImage[0][y][x];
//                }
//            }
//        }

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                double value = fakeImage[0][y][x];
                double normalizedValue = (value + 1) / 2.0; // (value - min) / (max - min);
                double brightness = normalizedValue * 255.0;
                int grayValue = (int) brightness;
                int rgb = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
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

    public double[][][] mean_1st_layer(double[][][][] array) {
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
        } catch (IOException e) {
            System.err.println("Error saving image: " + e.getMessage());
        }
    }
}