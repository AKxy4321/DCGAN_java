package DCGAN.test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
//import logger class
import java.util.logging.Level;
import java.util.logging.Logger;

public class DCGAN_ImplementationVariableSizeTesting {

    public static void main(String[] args) throws IOException {
        Logger logger = Logger.getLogger(DCGAN_Implementation.class.getName());
//        logger.setLevel(Level.OFF);
        int train_size = 320;
        int label = 0;
        double learning_rate_gen = 0.001F;
        double learning_rate_disc = 0.0001F;

        int inputSize = 100,
//            outputHeight1 = 6, outputWidth1=6, numFiltersPrev1=128, numFilters1=64,
//            filterSize1=3, filterSize2=3, numFilters2=32, stride1=2, stride2=2;
                outputHeight1 =7, outputWidth1=7, numFiltersPrev1=128, numFilters1=64,
                filterSize1=7, filterSize2=16, numFilters2=64, stride1=1, stride2=1;

        Discriminator_Implementation discriminator = new Discriminator_Implementation();

        Generator_Implementation generator = new Generator_Implementation(inputSize, outputHeight1, outputWidth1, numFiltersPrev1, numFilters1,
                filterSize1, filterSize2, numFilters2, stride1, stride2);
        UTIL UTIL = new UTIL();
        double[][][] realImages = new double[train_size][28][28];
        int[] index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < train_size; i++) {                  //load real images
            if (label > 9) {
                label = 0;
            }
            BufferedImage img = DCGAN.UTIL.mnist_load_index(label, index[0]);
            realImages[i] = DCGAN.UTIL.img_to_mat(img);

            // bring to range -1 to +1 from 0 to 1
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    realImages[i][y][x] = (realImages[i][y][x] * 2) - 1;
                }
            }

            //pretty print realImages[0]
            if (i == 0)
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        System.out.printf("%.2f ", realImages[i][y][x]);
                    }
                    System.out.println();
                }

            label++;
        }

        for (int i = 0; i < train_size; i++) {
            // GEN FORWARD
            double[] noise = new double[inputSize];
            Random rand = new Random();
            for (int z = 0; z < inputSize; z++) { //100
                noise[z] = rand.nextDouble();
            }
            System.out.println("Generator Forward");
            double[] gen_dense_output = generator.dense.forward(noise);
            logger.log(Level.INFO, "gen_dense_shape:" + gen_dense_output.length);
            double[] gen_batch1_output = generator.batch1.forward(gen_dense_output, true);
            double[][][] gen_batch1_output_unflattened = UTIL.unflatten(gen_batch1_output, numFilters1, outputHeight1, outputWidth1); //128, 7, 7);
//            logger.log(Level.INFO, "gen_dense_output_unflattened : " + Arrays.deepToString(gen_batch1_output_unflattened));
            logger.log(Level.INFO, "gen_batch1_output_unflattened shape:" + gen_batch1_output_unflattened.length + " " + gen_batch1_output_unflattened[0].length + " " + gen_batch1_output_unflattened[0][0].length);
            double[][][] gen_leakyrelu_output1 = generator.leakyReLU1.forward(gen_batch1_output_unflattened);

            double[][][] outputTconv1 = generator.tconv1.forward(gen_leakyrelu_output1);//1,128,13,13
            logger.log(Level.INFO, "outputTconv1 shape:" + outputTconv1.length + " " + outputTconv1[0].length + " " + outputTconv1[0][0].length);
            double[] gen_batch2_output = generator.batch2.forward(UTIL.flatten(outputTconv1), true);
            double[][][] gen_batch2_output_unflattened = UTIL.unflatten(gen_batch2_output, outputTconv1.length, outputTconv1[0].length, outputTconv1[0][0].length);

            double[][][] disc_leakyrelu_output2 = generator.leakyReLU2.forward(gen_batch2_output_unflattened);
            double[][][] outputTconv2 = generator.tconv2.forward(disc_leakyrelu_output2);
            logger.log(Level.INFO, "outputTconv2 shape:" + outputTconv2.length + " " + outputTconv2[0].length + " " + outputTconv2[0][0].length);
            double[][][] fakeImage = generator.tanh.forward(outputTconv2);
            System.out.printf("fakeImage depth %d length %d width %d\n", fakeImage.length, fakeImage[0].length, fakeImage[0][0].length);

            //DISC FORWARD REAL
            System.out.println("Discriminator Forward Real");
            double[][] real_output1 = discriminator.conv1.forward(realImages[i]);
            double[][] real_output1_2 = discriminator.leakyReLULayer1.forward(real_output1);
            double[][] real_output2 = discriminator.conv2.forward(real_output1_2);
            double[][] real_output2_2 = discriminator.leakyReLULayer2.forward(real_output2);
            double[] real_output2_2_flattened = UTIL.flatten(real_output2_2);
            double[] real_output_dense = discriminator.dense.forward(real_output2_2_flattened);
            double[] real_output_l = discriminator.sigmoidLayer.forward(real_output_dense);
            double[] discriminator_output_real = real_output_l;//

            //DISC FORWARD FAKE
            System.out.println("Discriminator Forward Fake");
            double[][] fake_output1 = discriminator.conv1.forward(fakeImage[0]);
            double[][] fake_output1_2 = discriminator.leakyReLULayer1.forward(fake_output1);
            double[][] fake_output2 = discriminator.conv2.forward(fake_output1_2);
            double[][] fake_output2_2 = discriminator.leakyReLULayer2.forward(fake_output2);
            double[] fake_output2_2_flattened = UTIL.flatten(fake_output2_2);
            double[] fake_out_dense = discriminator.dense.forward(fake_output2_2_flattened);
            logger.log(Level.INFO, "fake_out_dense : " + fake_out_dense[0]);
            double[] fake_output_l = discriminator.sigmoidLayer.forward(fake_out_dense);
            double[] discriminator_output_fake = fake_output_l;

            System.out.println("discriminator_output_fake : " + Arrays.toString(discriminator_output_fake));
            System.out.println("real_output_l : " + Arrays.toString(discriminator_output_real));

            // Calculate Loss
            double gen_loss = UTIL.gen_loss(discriminator_output_fake);
            double disc_loss = UTIL.disc_loss(discriminator_output_real, discriminator_output_fake);

            System.out.println("Gen_Loss " + gen_loss);
            System.out.println("Disc_Loss " + disc_loss);


            // DISC backward last layer
//            double disc_gradient = computeGradientDiscriminator(discriminator_output_real, discriminator_output_fake);
//            double[] disc_gradient_1d = com;//new double[]{disc_gradient}; //converting into array

            // DISC REAL BACKWARD
            System.out.println("Discriminator Backward Real");

            double[] real_gradient_sigmoid = discriminator.sigmoidLayer.backward(computeGradientReal(discriminator_output_real));//disc_gradient_1d);

            double[] real_gradient_dense = discriminator.dense.backward(real_gradient_sigmoid, learning_rate_disc);
            System.out.printf("Real Gradient Length %d\n", real_gradient_dense.length);
            int size = (int) Math.sqrt((double) real_gradient_dense.length / discriminator.conv2.filters.length);
            double[][] real_gradient_dense_unflattened = UTIL.unflatten(real_gradient_dense, discriminator.conv2.filters.length, size * size);
            double[][] real_gradient_leakyrelu2 = discriminator.leakyReLULayer2.backward(real_gradient_dense_unflattened);
            double[][] real_gradient_conv2 = discriminator.conv2.backward(real_gradient_leakyrelu2, learning_rate_disc);
            double[][] real_gradient_leakyrelu1 = discriminator.leakyReLULayer1.backward(real_gradient_conv2);
            double[][] real_gradient_conv1 = discriminator.conv1.backward(real_gradient_leakyrelu1, learning_rate_disc);
            double[][] real_gradient_generator = real_gradient_conv1;

            // DISC FAKE BACKWARD
            System.out.println("Discriminator Backward Fake");

            double[] fake_gradient_sigmoid = discriminator.sigmoidLayer.backward(computeGradientFake(discriminator_output_fake));
            double[] fake_gradient_dense = discriminator.dense.backward(fake_gradient_sigmoid, learning_rate_disc);
            double[][] fake_gradient_dense_unflattened = UTIL.unflatten(fake_gradient_dense, discriminator.conv2.filters.length, size * size);
            double[][] fake_gradient_leakyrelu2 = discriminator.leakyReLULayer2.backward(fake_gradient_dense_unflattened);
            double[][] fake_gradient_conv2 = discriminator.conv2.backward(fake_gradient_leakyrelu2, learning_rate_disc);
            double[][] fake_gradient_leakyrelu1 = discriminator.leakyReLULayer1.backward(fake_gradient_conv2);
            double[][] fake_gradient_conv1 = discriminator.conv1.backward(fake_gradient_leakyrelu1, learning_rate_disc);
            double[][] fake_gradient_discriminator = fake_gradient_conv1;

            // GEN BACKWARD
            System.out.println("Generator Backward");
            //since we want to ascend the gradient of the generator, we will negate fake_gradient_discriminator
//            for (int idx = 0; idx < fake_gradient_discriminator.length; idx++) {
//                for (int jdx = 0; jdx < fake_gradient_discriminator[0].length; jdx++) {
//                    fake_gradient_discriminator[idx][jdx] = -fake_gradient_discriminator[idx][jdx];
//                }
//            }
//            double[] fake_gradient = fake_gradient_discriminator; //UTIL.flatten(fakeImage[0]);
//            fake_gradient = computeGradientGenerator(discriminator_output_fake, fake_gradient_sigmoid, );
            double[][][] fake_back_gradient = new double[][][]{fake_gradient_discriminator}; //UTIL.unflatten(fake_gradient, 28, 28)}; // we want a 3d array, with only 1 channel
            double[][][] gradient0_1 = generator.tanh.backward(fake_back_gradient);
            double[][][] gradient1 = generator.tconv2.backward(gradient0_1, learning_rate_gen);
            double[][][] gradient1_2 = generator.leakyReLU2.backward(gradient1);
            double[][][] gradient1_3 = UTIL.unflatten(
                    generator.batch2.backward(UTIL.flatten(gradient1_2), learning_rate_gen),
                    gradient1_2.length, gradient1_2[0].length, gradient1_2[0][0].length);
            double[][][] gradient2 = generator.tconv1.backward(gradient1_3, learning_rate_gen);
            double[][][] gradient2_2 = generator.leakyReLU1.backward(gradient2);
            double[] out = UTIL.flatten(gradient2_2);
            double[] gradient3 = generator.batch1.backward(out, learning_rate_gen);
            generator.dense.backward(gradient3, learning_rate_gen);

            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    double value = fakeImage[0][y][x];
                    int brightness = (int) ((value + 1) * 0.5 * 255.0f);
                    if (y == 0 && x == 0)
                        logger.log(Level.INFO, "value : " + value + " brightness: " + brightness);
                    image.setRGB(x, y, new Color(brightness, brightness, brightness).getRGB());

                }
            }

            // Save the BufferedImage to a file
            File outputImageFile = new File("generated_image.png");
            try {
                ImageIO.write(image, "png", outputImageFile);
                System.out.println("Image saved successfully to: " + outputImageFile.getAbsolutePath());
            } catch (IOException e) {
                System.err.println("Error saving image: " + e.getMessage());
            }
        }
    }

    //    public static double computeGradientDiscriminator(double[] real_output, double[] fake_output) {
    //        double gradient = 0;
    //        for (int i = 0; i < real_output.length; i++) {
    //            gradient += 1.0 / (real_output[i] + UTIL.epsilon) - (fake_output[i] / (1.0 - fake_output[i] + UTIL.epsilon));
    //        }
    //        gradient /= real_output.length;
    //        return gradient;
    //    }

    public static double[] computeGradientReal(double[] real_output) {
        final double epsilon = 0.000000001;
        double[] gradient = new double[real_output.length];
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] = (+1 / (real_output[i] + epsilon)) + 0;// ((0) / (1 - real_output[i]));
        }
        return gradient;
    }

    public static double[] computeGradientFake(double[] fake_output) {
        final double epsilon = 0.000000001;

        double[] gradient = new double[fake_output.length];
        for (int i = 0; i < fake_output.length; i++) {
            // (-0 / fake_output[i]) +
            gradient[i] = ((-1) / (1 - fake_output[i] + epsilon));
        }
        return gradient;
    }

//    public static double[] computeGradientReal(double[] real_output) {
//        double[] gradient = new double[real_output.length];
//        for (int i = 0; i < real_output.length; i++) {
//            gradient[i] = real_output[i] - 1;
//        }
//        return gradient;
//    }
//
//    public static double[] computeGradientFake(double[] fake_output) {
//        double[] gradient = new double[fake_output.length];
//        for (int i = 0; i < fake_output.length; i++) {
//            gradient[i] = fake_output[i];
//        }
//        return gradient;
//    }
}

class Discriminator_Implementation {
    //output_size = (input_size - filter_size) / stride + 1

    ConvolutionalLayer conv1;
    LeakyReLULayer leakyReLULayer1;
    ConvolutionalLayer conv2;
    LeakyReLULayer leakyReLULayer2;
    DenseLayer dense;
    SigmoidLayer sigmoidLayer;

    public Discriminator_Implementation() {
        this.conv1 = new ConvolutionalLayer(5, 64);
        this.leakyReLULayer1 = new LeakyReLULayer();
        this.conv2 = new ConvolutionalLayer(5, 128);
        this.leakyReLULayer2 = new LeakyReLULayer();
        this.dense = new DenseLayer(24 * 24 * 128, 1);
        this.sigmoidLayer = new SigmoidLayer();
    }
}

class Generator_Implementation {
    //output_size = (input_size - 1) * stride + kernel_size - 2 * padding + output_padding
    //output_size = (input_size - 1) * stride + kernel_size (assuming no padding)
    int dense_output_size;

    DenseLayer dense;
    BatchNormalization batch1;
    LeakyReLULayer leakyReLU1;
    TransposeConvolutionalLayer tconv1;
    BatchNormalization batch2;
    LeakyReLULayer leakyReLU2;
    TransposeConvolutionalLayer tconv2;
    TanhLayer tanh;

    public Generator_Implementation(int inputSize, int outputHeight1, int outputWidth1, int numFiltersPrev1, int numFilters1, int filterSize1, int filterSize2, int numFilters2, int stride1, int stride2) {
        this.dense_output_size = outputHeight1 * outputWidth1 * numFiltersPrev1;

        this.dense = new DenseLayer(inputSize, this.dense_output_size);
        this.batch1 = new BatchNormalization();
        this.leakyReLU1 = new LeakyReLULayer();
        this.tconv1 = new TransposeConvolutionalLayer(numFiltersPrev1, filterSize1, numFilters1, stride1);
        this.batch2 = new BatchNormalization();
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(numFilters1, filterSize2, numFilters2, stride2);
        this.tanh = new TanhLayer();
    }

    public Generator_Implementation() {
        this.dense_output_size = 7 * 7 * 128;

        this.dense = new DenseLayer(100, this.dense_output_size);
        this.batch1 = new BatchNormalization();
        this.leakyReLU1 = new LeakyReLULayer();
        this.tconv1 = new TransposeConvolutionalLayer(128, 7, 64, 1);
        this.batch2 = new BatchNormalization();
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(64, 13, 32, 1);
        this.tanh = new TanhLayer();
    }
}

class UTIL {
    public static double epsilon = 1e-15F;

    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }

    public double[][][] unflatten(double[] input, int numFilters, int outputHeight, int outputWidth) {
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

    public static BufferedImage mnist_load_index(int label, int index) throws IOException {
        String mnistPath = Paths.get(".", "misc", "CNN", "data", "mnist_png", "mnist_png", "training").toString();
        File dir = new File(mnistPath, String.valueOf(label));
        String[] files = dir.list();
        assert files != null;
        String finalPath = mnistPath + File.separator + label + File.separator + files[index];
        return load_image(finalPath);
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
        double epsilon = 1e-15F; // small value to prevent taking log of zero
        for (int i = 0; i < y_true.length; i++) {
            double pred = (double) Math.max(Math.min(y_pred[i], 1. - epsilon), epsilon); // clamp predictions to avoid log(0)
            sum += (double) (-y_true[i] * Math.log(pred) - (1 - y_true[i]) * Math.log(1 - pred));
        }
        return sum / y_true.length;
    }

    public double[][] unflatten(double[] out_l, int height, int width) {

        double[][] output = new double[height][width];
        int k = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[i][j] = out_l[k++];
            }
        }
        return output;
    }

    public double[] flatten(double[][] input) {
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

    public double[] flatten(double[][][] input) {
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
        double new_array[] = new double[array.length];
        for (int i = 0; i < array.length; i++)
            new_array[i] = array[i] * -1;
        return new_array;
    }

    public static double[][] negate(double[][] array) {
        double[][] new_array = new double[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                new_array[i][j] = array[i][j] * -1;
        return new_array;
    }

    public static double[][][] negate(double[][][] array) {
        double[][][] new_array = new double[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    new_array[i][j][k] = array[i][j][k] * -1;
        return new_array;
    }

//    public double[] computeGradientReal(double[] real_output) {
//        double[] gradient = new double[real_output.length];
//        for (int i = 0; i < real_output.length; i++) {
//            gradient[i] = real_output[i] - 1;
//        }
//        return gradient;
//    }
//
//    public double[] computeGradientFake(double[] fake_output) {
//        double[] gradient = new double[fake_output.length];
//        for (int i = 0; i < fake_output.length; i++) {
//            gradient[i] = fake_output[i];
//        }
//        return gradient;
//    }
}