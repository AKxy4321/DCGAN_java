package DCGAN;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DCGAN_Implementation {

    public static void main(String[] args) throws IOException {
        Logger logger = Logger.getLogger(DCGAN_Implementation.class.getName());
        int train_size = 25000;
        int label = 0;
        double learning_rate_gen = 1e-8;
        double learning_rate_disc = 1e-8;
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        Generator_Implementation generator = new Generator_Implementation();
        UTIL UTIL = new UTIL();
        System.out.println("Loading Images");
        double[][][] realImages = new double[train_size][28][28];
        int[] index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < train_size; i++) {
            if (label > 9) {
                label = 0;
            }
            BufferedImage img = DCGAN.UTIL.mnist_load_index(label, index[label]);
            realImages[i] = DCGAN.UTIL.img_to_mat(img);
            label++;
        }

        for (int i = 0; i < train_size; i++) {
            double[] noise = XavierInitializer.xavierInit1D(100);

            System.out.println("Generator Forward");
            double[] gen_dense_output = generator.dense.forward(noise);
            double[] gen_batch1_output = generator.batch1.forward(gen_dense_output, true);
            double[][][] gen_batch1_output_unflattened = UTIL.unflatten(gen_batch1_output, 128, 7, 7);
            double[][][] gen_leakyrelu_output1 = generator.leakyReLU1.forward(gen_batch1_output_unflattened);
            double[][][] outputTconv1 = generator.tconv1.forward(gen_leakyrelu_output1);
            double[] gen_batch2_output = generator.batch2.forward(UTIL.flatten(outputTconv1), true);
            double[][][] gen_batch2_output_unflattened = UTIL.unflatten(gen_batch2_output, outputTconv1.length, outputTconv1[0].length, outputTconv1[0][0].length);
            double[][][] gen_leakyrelu_output2 = generator.leakyReLU2.forward(gen_batch2_output_unflattened);
            double[][][] outputTconv2 = generator.tconv2.forward(gen_leakyrelu_output2);
            double[] gen_batch3_output = generator.batch3.forward(UTIL.flatten(outputTconv2), true);
            double[][][] gen_batch3_output_unflattened = UTIL.unflatten(gen_batch3_output, outputTconv2.length, outputTconv2[0].length, outputTconv2[0][0].length);
            double[][][] gen_leakyrelu_output3 = generator.leakyReLU3.forward(gen_batch3_output_unflattened);
            double[][][] fakeImage = generator.tanh.forward(gen_leakyrelu_output3);

            System.out.printf("fakeImage depth %d length %d width %d\n", fakeImage.length, fakeImage[0].length, fakeImage[0][0].length);

            System.out.println("Discriminator Forward Real");
            double[][] real_output1 = discriminator.conv1.forward(realImages[i]);
            double[][] real_output1_2 = discriminator.leakyReLULayer1.forward(real_output1);
            double[][] real_output2 = discriminator.conv2.forward(real_output1_2);
            double[][] real_output2_2 = discriminator.leakyReLULayer2.forward(real_output2);
            double[] real_output2_2_flattened = UTIL.flatten(real_output2_2);
            double[] real_output_dense = discriminator.dense.forward(real_output2_2_flattened);
            double[] discriminator_output_real = discriminator.sigmoidLayer.forward(real_output_dense);

            System.out.println("Discriminator Forward Fake");
            double[][] fake_output1 = discriminator.conv1.forward(fakeImage[0]);
            double[][] fake_output1_2 = discriminator.leakyReLULayer1.forward(fake_output1);
            double[][] fake_output2 = discriminator.conv2.forward(fake_output1_2);
            double[][] fake_output2_2 = discriminator.leakyReLULayer2.forward(fake_output2);
            double[] fake_output2_2_flattened = UTIL.flatten(fake_output2_2);
            double[] fake_out_dense = discriminator.dense.forward(fake_output2_2_flattened);
            logger.log(Level.INFO, "fake_out_dense : " + fake_out_dense[0]);
            double[] discriminator_output_fake = discriminator.sigmoidLayer.forward(fake_out_dense);

            double gen_loss = UTIL.gen_loss(discriminator_output_fake);
            double disc_loss = UTIL.disc_loss(discriminator_output_real, discriminator_output_fake);

            System.out.println("Gen_Loss " + gen_loss);
            System.out.println("Disc_Loss " + disc_loss);

            System.out.println("Discriminator Backward");
            double[] dg = computeGradientDiscriminator(discriminator_output_real, discriminator_output_fake);
            double[] disc_gradient = DCGAN.UTIL.negate(dg);
            double[] disc_gradient_sigmoid = discriminator.sigmoidLayer.backward(disc_gradient);
            double[] disc_gradient_dense = discriminator.dense.backward(disc_gradient_sigmoid, learning_rate_disc);
            int size = (int) Math.sqrt((double) disc_gradient_dense.length / discriminator.conv2.filters.length);
            double[][] disc_gradient_dense_unflattened = UTIL.unflatten(disc_gradient_dense, discriminator.conv2.filters.length, size * size);
            double[][] disc_gradient_leakyrelu2 = discriminator.leakyReLULayer2.backward(disc_gradient_dense_unflattened);
            double[][] disc_gradient_conv2 = discriminator.conv2.backward(disc_gradient_leakyrelu2, learning_rate_disc);
            double[][] disc_gradient_leakyrelu1 = discriminator.leakyReLULayer1.backward(disc_gradient_conv2);
            double[][] disc_g = discriminator.conv1.backward(disc_gradient_leakyrelu1, learning_rate_disc);

            System.out.println("Generator Backward");
            double[][][] fake_back_gradient = new double[][][]{disc_g};
            double[][][] gradient0 = generator.tanh.backward(fake_back_gradient);
            double[][][] gradient0_1 = generator.leakyReLU3.backward(gradient0);
            double[][][] gradient0_2 = UTIL.unflatten(generator.batch3.backward(UTIL.flatten(gradient0_1), learning_rate_gen),
            gradient0_1.length, gradient0_1[0].length, gradient0_1[0][0].length);
            double[][][] gradient1 = generator.tconv2.backward(gradient0_2, learning_rate_gen);
            double[][][] gradient1_2 = generator.leakyReLU2.backward(gradient1);
            double[][][] gradient1_3 = UTIL.unflatten(
            generator.batch2.backward(UTIL.flatten(gradient1_2), learning_rate_gen),
            gradient1_2.length, gradient1_2[0].length, gradient1_2[0][0].length);
            double[][][] gradient2 = generator.tconv1.backward(gradient1_3, learning_rate_gen);
            double[][][] gradient2_2 = generator.leakyReLU1.backward(gradient2);
            double[] out = UTIL.flatten(gradient2_2);
            double[] gradient3 = generator.batch1.backward(out, learning_rate_gen);
            generator.dense.backward(gradient3, learning_rate_gen);

            BufferedImage image = getBufferedImage(fakeImage);

            File outputImageFile = new File("generated_image.png");
            try {
                ImageIO.write(image, "png", outputImageFile);
                System.out.println("Image saved successfully to: " + outputImageFile.getAbsolutePath());
            } catch (IOException e) {
                System.err.println("Error saving image: " + e.getMessage());
            }
        }
    }

    private static BufferedImage getBufferedImage(double[][][] fakeImage) {
        double max = fakeImage[0][0][0];
        double min = fakeImage[0][0][0];
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                if (max < fakeImage[0][y][x]) {
                    max = fakeImage[0][y][x];
                }
                if (min > fakeImage[0][y][x]) {
                    min = fakeImage[0][y][x];
                }
            }
        }

        for (int y = 0 ; y < 28 ; y++) {
            for (int x = 0 ; x < 28 ; x++) {
                double value = fakeImage[0][y][x];
                double normalizedValue = (value - min) / (max - min);
                double brightness = normalizedValue * 255.0;
                int grayValue = (int) brightness;
                int rgb = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    public static double[] computeGradientDiscriminator(double[] real_output, double[] fake_output) {
        double[] gradient = new double[real_output.length];
        double epsilon = 1e-3;
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] += 1.0 / (real_output[i] + epsilon) - (1 / (1.0 - fake_output[i] + epsilon));
        }
        return gradient;
    }

}

class Discriminator_Implementation {
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
    int dense_output_size;
    DenseLayer dense;
    BatchNormalization batch1;
    LeakyReLULayer leakyReLU1;
    TransposeConvolutionalLayer tconv1;
    BatchNormalization batch2;
    LeakyReLULayer leakyReLU2;
    TransposeConvolutionalLayer tconv2;
    BatchNormalization batch3;
    LeakyReLULayer leakyReLU3;
    TanhLayer tanh;

    public Generator_Implementation() {
        this.dense_output_size = 7 * 7 * 128;
        this.dense = new DenseLayer(100, this.dense_output_size);
        this.batch1 = new BatchNormalization();
        this.leakyReLU1 = new LeakyReLULayer();
        this.tconv1 = new TransposeConvolutionalLayer(128, 5, 64, 1);
        this.batch2 = new BatchNormalization();
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(64, 8, 1, 2);
        this.batch3 = new BatchNormalization();
        this.leakyReLU3 = new LeakyReLULayer();
        this.tanh = new TanhLayer();
    }
}

class UTIL {
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
        String mnistPath = Paths.get(".", "data", "mnist_png", "mnist_png", "training").toString();
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
        double epsilon = 1e-3F;
        for (int i = 0; i < y_true.length; i++) {
            double pred = Math.max(Math.min(y_pred[i], 1. - epsilon), epsilon);
            sum += -y_true[i] * Math.log(pred) - (1 - y_true[i]) * Math.log(1 - pred);
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

    public static double[] negate(double[] array){
        double[] new_array = new double[array.length];
        for(int i=0;i<array.length;i++)
            new_array[i] = array[i] * -1;
        return new_array;
    }

}
