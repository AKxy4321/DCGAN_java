package DCGAN;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DCGAN_Implementation {

    public static void main(String[] args) {
        DCGAN_Implementation dcgan = new DCGAN_Implementation();
//        dcgan.dcgan_execute();
        dcgan.discriminator_execute();
    }

    public void discriminator_execute() {
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        int num_images = 200;
        int num_images_test = 100;
        int num_epochs = 100;

        double[][][] fakeImages_train = new double[num_images][28][28];
        double[][][] realImages_train = new double[num_images][28][28];

        double[][][] fakeImages_test = new double[num_images_test][28][28];
        double[][][] realImages_test = new double[num_images_test][28][28];

        for (int i = 0; i < num_images; i++) {
            System.out.println(i);
            realImages_train[i] = DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(1, i));
            fakeImages_train[i] = DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(0, i));
        }

        for (int i = 0; i < num_images_test; i++) {
            realImages_test[i] = DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(1, i + num_images));
            fakeImages_test[i] = DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(0, i + num_images));
        }

        for (int epoch = 0; epoch < num_epochs; epoch++) {
            double total_loss = 0.0;

            double[][] outputGradients = new double[num_images][1];
            // Train discriminator to identify fake from real images
            double[][][] avg_image = new double[1][28][28];
            for (int i = 0; i < num_images; i++) {

                double[] real_output = discriminator.getOutput(realImages_train[i]);
//                System.out.println(Arrays.toString(real_output));
                double[] fake_output = discriminator.getOutput(fakeImages_train[i]);
                double loss = lossDiscriminatorMSE(real_output, fake_output);
                double[] gradient = gradientDiscriminatorMSE(real_output, fake_output);
                total_loss += loss;

                // Update gradients
                outputGradients[i] = gradient;


                // update avg_image
                for (int j = 0; j < 28; j++) {
                    for (int k = 0; k < 28; k++) {
                        avg_image[0][j][k] += fakeImages_train[i][j][k]+realImages_train[i][j][k];
                    }
                }
            }

            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    avg_image[0][j][k] /= (realImages_train.length*2);
                }
            }
            discriminator.getOutput(avg_image[0]);

            // Update discriminator parameters after processing all images
            discriminator.updateParameters(UTIL.mean_1st_layer(outputGradients), 0.0005);

            // Calculate test loss and accuracy
            double test_loss = 0.0;
            double accuracy = 0.0;
            for (int i = 0; i < num_images_test; i++) {
                double[] test_real_outputs = discriminator.getOutput(realImages_test[i]);
                double[] test_fake_outputs = discriminator.getOutput(fakeImages_test[i]);
                test_loss += lossDiscriminatorMSE(test_real_outputs, test_fake_outputs);
                accuracy += calculateAccuracy(test_real_outputs, test_fake_outputs);
                if (epoch == num_epochs - 1) {
                    System.out.println("Real output: " + Arrays.toString(test_real_outputs) + ", Fake output: " + Arrays.toString(test_fake_outputs));
                }
            }
            test_loss /= num_images_test;
            accuracy /= (2 * num_images_test);

            System.out.println("Epoch: " + (epoch + 1) + ", Average Training Loss: " + (total_loss / num_images) +
                    ", Test Loss: " + test_loss + ", Test Accuracy: " + accuracy + ", Total training loss: " + total_loss);


        }
    }

    public double lossDiscriminatorMSE(double[] real_output, double[] fake_output) {
        double loss = 0;
        for (int i = 0; i < real_output.length; i++) {
            loss += Math.pow(real_output[i] - 1, 2) + Math.pow(fake_output[i], 2);
        }
        return loss / real_output.length;
    }

    public double[] gradientDiscriminatorMSE(double[] real_output, double[] fake_output) {
        double[] gradient = new double[real_output.length];
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] = 2 * (real_output[i] - 1) + 2 * fake_output[i];
        }
        return gradient;
    }

    public double calculateAccuracy(double[] real_outputs, double[] fake_outputs) {
        double accuracy = 0;
        for (int j = 0; j < real_outputs.length; j++) {
            if (real_outputs[j] > 0.5) {
                accuracy += 1;
            }
            if (fake_outputs[j] < 0.5) {
                accuracy += 1;
            }
        }
        return accuracy;
    }


    public void dcgan_execute() {
        Logger logger = Logger.getLogger(DCGAN_Implementation.class.getName());
        int train_size = 3200;
        int label = 0;
        double learning_rate_gen = 1e-2;
        double learning_rate_disc = 1e-2;
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        Generator_Implementation generator = new Generator_Implementation();
        UTIL UTIL = new UTIL();
        System.out.println("Loading Images");

        int batch_size = 8;
        // minibatch gradient descent
        for (int epochs = 0; epochs < 50; epochs++) {
            int[] index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            logger.log(Level.INFO, "Epoch " + epochs);
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {
                double[][][][] fakeImages = new double[batch_size][1][28][28];
                double[][][][] realImages = new double[batch_size][1][28][28];
                for (int real_idx = 0; real_idx < batch_size; real_idx++) {
                    BufferedImage img = DCGAN.UTIL.mnist_load_index(0, index[0]);
                    realImages[real_idx] = new double[][][]{DCGAN.UTIL.img_to_mat(img)};
                    index[0] += 1;
                }
                for (int j = 0; j < batch_size; j++) {
                    fakeImages[j] = generator.generateImage();
                    System.out.print(j + " ");
                }
                System.out.println("Finished loading images");
                double[][] discriminator_fake_outputs = new double[batch_size][1];
                double[][] discriminator_real_outputs = new double[batch_size][1];
                double[] gen_losses = new double[batch_size];
                double[] disc_losses = new double[batch_size];


                for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                    System.out.println(realImages[img_idx][0].length + " " + realImages[img_idx][0][0].length);
                    double[] discriminator_output_real = discriminator.getOutput(realImages[img_idx][0]);

                    double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx][0]);

                    discriminator_real_outputs[img_idx] = discriminator_output_real;
                    discriminator_fake_outputs[img_idx] = discriminator_output_fake;

                    double gen_loss = UTIL.gen_loss(discriminator_output_fake);
                    double disc_loss = UTIL.disc_loss(discriminator_output_real, discriminator_output_fake);
                    gen_losses[img_idx] = gen_loss;
                    disc_losses[img_idx] = disc_loss;

                    System.out.print(img_idx + " ");
                }

                logger.log(Level.INFO, "Epoch:" + epochs + " batch:" + batch_idx);
                logger.log(Level.INFO, "Gen_Loss " + DCGAN.UTIL.mean(gen_losses));
                logger.log(Level.INFO, "Disc_Loss " + DCGAN.UTIL.mean(disc_losses));

                double[][] disc_output_gradients = new double[batch_size][1];
                double[][][][] gen_output_gradients = new double[batch_size][1][28][28];

                for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                    double[] discriminator_output_fake = discriminator_fake_outputs[img_idx];
                    double[] discriminator_output_real = discriminator_real_outputs[img_idx];

//                System.out.println("computing discriminator gradients");
                    // gradient wrt both real and fake output
                    double[] dg = computeGradientDiscriminator(discriminator_output_real, discriminator_output_fake);
//                    disc_output_gradients[img_idx] = UTIL.negate(dg);

                    //gradient wrt only fake output
                    double[] disc_gradient_fake = computeGradientDiscriminatorWRTFake(discriminator_output_fake);

                    // do one forward pass so that convolution layer stores this image
                    discriminator.getOutput(fakeImages[img_idx][0]);

                    double[][][] fake_back_gradient = discriminator.backward(disc_gradient_fake);
                    gen_output_gradients[img_idx] = fake_back_gradient;
                }

                // back prop with update parameters
                double[] disc_output_gradient = UTIL.mean_1st_layer(disc_output_gradients);
                double[][][] gen_output_gradient = UTIL.mean_1st_layer(gen_output_gradients);

                // update discriminator
                discriminator.updateParameters(disc_output_gradient, learning_rate_disc);


                // update generator
                generator.updateParameters(gen_output_gradient, learning_rate_gen);

                // generate image
                BufferedImage image = DCGAN.UTIL.getBufferedImage(generator.generateImage());

                DCGAN.UTIL.saveImage(image, "generated_image.png");
            }
        }
    }

    public static double lossDiscriminator(double[] real_output, double[] fake_output) {
        double loss = 0;
        for (int i = 0; i < real_output.length; i++) {
            loss += -Math.log(real_output[i]) - Math.log(1 - fake_output[i]);
        }
        return loss / real_output.length;
    }

    public static double[] computeGradientDiscriminator(double[] real_output, double[] fake_output) {
        double[] gradient = new double[real_output.length];
        double epsilon = 1e-1F;
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] += 1.0 / (real_output[i] + epsilon) - (1 / (1.0 - fake_output[i] + epsilon));
        }
        return gradient;
    }

    public static double[] computeGradientDiscriminatorWRTFake(double[] fake_output) {
        double[] gradient = new double[fake_output.length];
        double epsilon = 1e-1F;
        for (int i = 0; i < fake_output.length; i++) {
            gradient[i] += -(1 / (1.0 - fake_output[i] + epsilon));
        }
        return gradient;
    }

}

class Discriminator_Implementation {
    Convolution conv1;
    LeakyReLULayer leakyReLULayer1;
    DenseLayer dense;
    SigmoidLayer sigmoidLayer;

    public Discriminator_Implementation() {
        int inputWidth = 28, inputHeight = 28;
        this.conv1 = new Convolution(5, 64, 3, inputWidth, inputHeight, 1);
        this.leakyReLULayer1 = new LeakyReLULayer();
        this.dense = new DenseLayer(conv1.output_depth*conv1.output_width*conv1.output_height, 1);
        this.sigmoidLayer = new SigmoidLayer();
    }

    public double[] getOutput(double[][] img) {
        double[][][] input = new double[1][][];
        input[0] = img;
        double[][][] output_conv1 = this.conv1.forward(input);
        double[][][] output_leakyRELU1 = this.leakyReLULayer1.forward(output_conv1);
        double[] output_leakyRELU1_flattened = UTIL.flatten(output_leakyRELU1);

        double[] output_dense = this.dense.forward(output_leakyRELU1_flattened);
        double[] discriminator_output = this.sigmoidLayer.forward(output_dense);
        return discriminator_output;
    }

    public double[][][] backward(double[] outputGradient) {
        double[] disc_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_gradient_dense = this.dense.backward(disc_gradient_sigmoid);

        double[][][] disc_in_gradient_dense_unflattened = UTIL.unflatten(disc_gradient_dense, leakyReLULayer1.output.length, leakyReLULayer1.output[0].length, leakyReLULayer1.output[0][0].length);
        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_dense_unflattened);
        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);
        // now we have the gradient of the loss function for the generated output wrt to the generator output(which is nothing but dou J / dou image)
        return disc_in_gradient_conv1; // this is fake_back_gradient
    }

    public void updateParameters(double[] outputGradient, double learning_rate_disc) {
        double[] disc_in_gradient_sigmoid = this.sigmoidLayer.backward(outputGradient);
        double[] disc_in_gradient_dense = this.dense.backward(disc_in_gradient_sigmoid);

        this.dense.updateParameters(disc_in_gradient_sigmoid, learning_rate_disc);

        double[][][] disc_in_gradient_dense_unflattened = UTIL.unflatten(disc_in_gradient_dense, leakyReLULayer1.output.length, leakyReLULayer1.output[0].length, leakyReLULayer1.output[0][0].length);
        double[][][] disc_in_gradient_leakyrelu1 = this.leakyReLULayer1.backward(disc_in_gradient_dense_unflattened);
        double[][][] disc_in_gradient_conv1 = this.conv1.backprop(disc_in_gradient_leakyrelu1);

        conv1.updateParameters(disc_in_gradient_leakyrelu1, learning_rate_disc);
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
        this.tconv1 = new TransposeConvolutionalLayer(256, 5, 64, 1);
        this.batch2 = new BatchNormalization();
        this.leakyReLU2 = new LeakyReLULayer();
        this.tconv2 = new TransposeConvolutionalLayer(64, 8, 1, 2);
        this.batch3 = new BatchNormalization();
        this.leakyReLU3 = new LeakyReLULayer();
        this.tanh = new TanhLayer();
        // size of tanh is supposed to be 64x5*5
        /*
         * [128,1,6,6]   [3,3]   2
         * 128,1,13,13   [4,4]   2
         *  128,1,28,28
         * */

    }

    public double[][][] generateImage() {
        double[] noise = XavierInitializer.xavierInit1D(100); // generate noise input that we want to pass to the generator

        double[] gen_dense_output = this.dense.forward(noise);
        double[] gen_batch1_output = this.batch1.forward(gen_dense_output, true);
        double[][][] gen_batch1_output_unflattened = UTIL.unflatten(gen_batch1_output, 128, 7, 7);
        double[][][] gen_leakyrelu_output1 = this.leakyReLU1.forward(gen_batch1_output_unflattened);
        double[][][] outputTconv1 = this.tconv1.forward(gen_leakyrelu_output1);
        double[] gen_batch2_output = this.batch2.forward(UTIL.flatten(outputTconv1), true);
        double[][][] gen_batch2_output_unflattened = UTIL.unflatten(gen_batch2_output, outputTconv1.length, outputTconv1[0].length, outputTconv1[0][0].length);
        double[][][] gen_leakyrelu_output2 = this.leakyReLU2.forward(gen_batch2_output_unflattened);
        double[][][] outputTconv2 = this.tconv2.forward(gen_leakyrelu_output2);
        double[] gen_batch3_output = this.batch3.forward(UTIL.flatten(outputTconv2), true);
        double[][][] gen_batch3_output_unflattened = UTIL.unflatten(gen_batch3_output, outputTconv2.length, outputTconv2[0].length, outputTconv2[0][0].length);
        double[][][] gen_leakyrelu_output3 = this.leakyReLU3.forward(gen_batch3_output_unflattened);
        double[][][] fakeImage = this.tanh.forward(gen_leakyrelu_output3);
        return fakeImage;
    }

    public void updateParameters(double[][][] gen_output_gradient, double learning_rate_gen) {

        double[][][] gradient0 = this.tanh.backward(gen_output_gradient);
        double[][][] gradient0_1 = this.leakyReLU3.backward(gradient0);
        double[] gradient_0_1_flattened = UTIL.flatten(gradient0_1);
        double[][][] gradient0_2 = UTIL.unflatten(this.batch3.backward(gradient_0_1_flattened),
                gradient0_1.length, gradient0_1[0].length, gradient0_1[0][0].length);
        this.batch3.updateParameters(gradient_0_1_flattened, learning_rate_gen);

        double[][][] gradient1 = this.tconv2.backward(gradient0_2);
        this.tconv2.updateParameters(gradient0_2, learning_rate_gen);

        double[][][] gradient1_2 = this.leakyReLU2.backward(gradient1);
        double[] gradient1_2_flattened = UTIL.flatten(gradient1_2);
        double[][][] gradient1_3 = UTIL.unflatten(
                this.batch2.backward(gradient1_2_flattened),
                gradient1_2.length, gradient1_2[0].length, gradient1_2[0][0].length);
        this.batch2.updateParameters(gradient1_2_flattened, learning_rate_gen);

        double[][][] gradient2 = this.tconv1.backward(gradient1_3);
        this.tconv1.updateParameters(gradient1_3, learning_rate_gen);

        double[][][] gradient2_2 = this.leakyReLU1.backward(gradient2);
        double[] out = UTIL.flatten(gradient2_2);
        double[] gradient3 = this.batch1.backward(out);
        this.batch1.updateParameters(out, learning_rate_gen);

        double[] gradient4 = this.dense.backward(gradient3);
        this.dense.updateParameters(gradient3, learning_rate_gen);
    }
}

class UTIL {
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
            double pred = Math.max(Math.min(y_pred[i], 1. - epsilon), epsilon);
            sum += -y_true[i] * Math.log(pred) - (1 - y_true[i]) * Math.log(1 - pred);
        }
        return sum / y_true.length;
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