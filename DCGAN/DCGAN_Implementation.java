package DCGAN;

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

public class DCGAN_Implementation {

    public static void main(String[] args) throws IOException {
        Logger logger = Logger.getLogger(DCGAN_Implementation.class.getName());
        logger.setLevel(Level.OFF);
        int train_size = 100;
        int label = 0;
        double learning_rate_gen = 0.001F;
        double learning_rate_disc = 0.0001F;
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        Generator_Implementation generator = new Generator_Implementation();
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

        for (int i = 0; i < 10; i++) {
            // GEN FORWARD
            double[] noise = new double[100];
            Random rand = new Random();
            for (int z = 0; z < 100; z++) {
                noise[z] = rand.nextDouble();
            }
            System.out.println("Generator Forward");
            double[] gen_dense_output = generator.dense.forward(noise);
            logger.log(Level.INFO, "gen_dense_output : " + Arrays.toString(gen_dense_output));
            double[][][] gen_dense_output_unflattened = generator.tconv1.unflattenArray(gen_dense_output, 128, 7, 7);
            logger.log(Level.INFO, "gen_dense_output_unflattened : " + Arrays.deepToString(gen_dense_output_unflattened));
            double[][][] gen_leakyrelu_output1 = generator.leakyReLU1.forward(gen_dense_output_unflattened);
            double[][][] outputTconv1 = generator.tconv1.forward(gen_leakyrelu_output1);
            double[][][] disc_leakyrelu_output2 = generator.leakyReLU2.forward(outputTconv1);
            double[][][] outputTconv2 = generator.tconv2.forward(disc_leakyrelu_output2);
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

            //DISC FORWARD FAKE
            System.out.println("Discriminator Forward Fake");
            double[][] fake_output1 = discriminator.conv1.forward(fakeImage[0]);
            double[][] fake_output1_2 = discriminator.leakyReLULayer1.forward(fake_output1);
            double[][] fake_output2 = discriminator.conv2.forward(fake_output1_2);
            double[][] fake_output2_2 = discriminator.leakyReLULayer2.forward(fake_output2);
            double[] fake_output2_2_flattened = UTIL.flatten(fake_output2_2);
            double[] fake_out_dense = discriminator.dense.forward(fake_output2_2_flattened);
            double[] fake_output_l = discriminator.sigmoidLayer.forward(fake_out_dense);

            System.out.println("fake_output_l : " + Arrays.toString(fake_output_l));
            System.out.println("real_output_l : " + Arrays.toString(real_output_l));

            // Calculate Loss
            double gen_loss = UTIL.gen_loss(fake_output_l);
            double disc_loss = UTIL.disc_loss(real_output_l, fake_output_l);

            System.out.println("Gen_Loss " + gen_loss);
            System.out.println("Disc_Loss " + disc_loss);

            // GEN BACKWARD
            System.out.println("Generator Backward");
            double[] fake_gradient_dense = UTIL.flatten(fakeImage[0]);
            fake_gradient_dense = UTIL.computeGradientFake(fake_gradient_dense);
            double[][][] fake_back_gradient = new double[1][28][28];
            fake_back_gradient[0] = UTIL.unflatten(fake_gradient_dense, 28, 28);

            double[][][] gradient0_1 = generator.tanh.backward(fake_back_gradient);
            double[][][] gradient1 = generator.tconv2.backward(fake_back_gradient, learning_rate_gen);
            double[][][] gradient1_2 = generator.leakyReLU2.backward(gradient1);
            double[][][] gradient2 = generator.tconv1.backward(gradient1_2, learning_rate_gen);
            double[][][] gradient2_2 = generator.leakyReLU1.backward(gradient2);
            double[] out = UTIL.flatten(gradient2_2);
            generator.dense.backward(out, learning_rate_gen);

            // DISC REAL BACKWARD
            System.out.println("Discriminator Backward Real");
            double[] real_gradient_dense = UTIL.computeGradientReal(real_output_l);
            double[] real_gradient_sigmoid = discriminator.sigmoidLayer.backward(real_gradient_dense);

            real_gradient_dense = discriminator.dense.backward(real_gradient_sigmoid, learning_rate_disc);
            System.out.printf("Real Gradient Length %d\n", real_gradient_dense.length);
            int size = (int) Math.sqrt((double) real_gradient_dense.length / discriminator.conv2.filters.length);
            double[][] real_gradient_dense_unflattened = UTIL.unflatten(real_gradient_dense, discriminator.conv2.filters.length, size * size);
            double[][] real_gradient_leakyrelu2 = discriminator.leakyReLULayer2.backward(real_gradient_dense_unflattened);
            double[][] real_gradient_conv2 = discriminator.conv2.backward(real_gradient_leakyrelu2, learning_rate_disc);
            double[][] real_gradient_leakyrelu1 = discriminator.leakyReLULayer1.backward(real_gradient_conv2);
            double[][] real_gradient_conv1 = discriminator.conv1.backward(real_gradient_leakyrelu1, learning_rate_disc);

            // DISC FAKE BACKWARD
            System.out.println("Discriminator Backward Fake");
            double[] fake_gradient_l_1 = UTIL.computeGradientFake(fake_output_l);
            double[] fake_gradient_sigmoid = discriminator.sigmoidLayer.backward(fake_gradient_l_1);
            fake_gradient_dense = discriminator.dense.backward(fake_gradient_sigmoid, learning_rate_disc);
            double[][] fake_gradient_dense_unflattened = UTIL.unflatten(fake_gradient_dense, discriminator.conv2.filters.length, size * size);
            double[][] fake_gradient_leakyrelu2 = discriminator.leakyReLULayer2.backward(fake_gradient_dense_unflattened);
            double[][] fake_gradient_conv2 = discriminator.conv2.backward(fake_gradient_leakyrelu2, learning_rate_disc);
            double[][] fake_gradient_leakyrelu1 = discriminator.leakyReLULayer1.backward(fake_gradient_conv2);
            double[][] fake_gradient_conv1 = discriminator.conv1.backward(fake_gradient_leakyrelu1, learning_rate_disc);

            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    double value = fakeImage[0][y][x];
                    int brightness = (int) ((value + 1) * 0.5 * 255.0f);
                    if (y==0 && x==0)
                        logger.log(Level.INFO,"value : "+value+ " brightness: " + brightness);
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
    LeakyReLULayer leakyReLU1;
    TransposeConvolutionalLayer tconv1;
    LeakyReLULayer leakyReLU2;
    TransposeConvolutionalLayer tconv2;
    TanhLayer tanh;

    public Generator_Implementation() {
        this.dense_output_size = 7 * 7 * 128;

        this.dense = new DenseLayer(100, this.dense_output_size);
        this.leakyReLU1 = new LeakyReLULayer();
        this.tconv1 = new TransposeConvolutionalLayer(128, 7, 64, 1);
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(64, 16, 1, 1);
        this.tanh = new TanhLayer();
    }
}

class UTIL {
    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
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

    public double[] computeGradientReal(double[] real_output) {
        double[] gradient = new double[real_output.length];
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] = 1 / real_output[i];
        }
        return gradient;
    }

    public double[] computeGradientFake(double[] fake_output) {
        double[] gradient = new double[fake_output.length];
        for (int i = 0; i < fake_output.length; i++) {
            gradient[i] = 1 / fake_output[i] - 1;
        }
        return gradient;
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