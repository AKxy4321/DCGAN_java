package DCGAN;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;

public class DCGAN_Implementation {
    public static void main(String[] args) throws IOException {
        int train_size = 100;
        int label = 0;
        float learning_rate_gen = 0.01F;
        float learning_rate_disc = 0.000001F;
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        Generator_Implementation generator = new Generator_Implementation();
        UTIL UTIL = new UTIL();
        float[][][] realImages = new float[train_size][28][28];
        int[] index= {0,0,0,0,0,0,0,0,0,0};
        for(int i = 0 ; i < train_size ; i++) {                  //load real images
            if (label > 9) {
                label = 0;
            }
            BufferedImage img = DCGAN.UTIL.mnist_load_index(label, index[0]);
            realImages[i] = DCGAN.UTIL.img_to_mat(img);
            label++;
        }

        for(int i = 0 ; i < train_size ; i++) {
            // GEN FORWARD
            float[] noise = new float[100];
            Random rand = new Random();
            for (int z = 0; z < 100; z++) {
                noise[z] = rand.nextFloat();
            }
            System.out.println("Generator Forward");
            float[] out_l = generator.dense.forward_ReLU(noise);
            float[][][] output_l = generator.tconv1.unflattenArray(out_l, 128, 7, 7);
            float[][][] output1 = generator.tconv1.forward_ReLU(output_l);
            float[][][] fakeImage = generator.tconv2.forward_tanh(output1);

            System.out.printf("fakeImage depth %d length %d width %d\n", fakeImage.length, fakeImage[0].length, fakeImage[0][0].length);

            //DISC FORWARD REAL
            System.out.println("Discriminator Forward Real");
            float[][] real_output1 = discriminator.conv1.forward(realImages[i]);
            float[][] real_output2 = discriminator.conv2.forward(real_output1);
            float[] real_out_l = UTIL.flatten(real_output2);
            float[] real_output_l = discriminator.dense.forward_Sigmoid(real_out_l);

            //DISC FORWARD FAKE
            System.out.println("Discriminator Forward Fake");
            float[][] fake_output1 = discriminator.conv1.forward(fakeImage[0]);
            float[][] fake_output2 = discriminator.conv2.forward(fake_output1);
            float[] fake_out_l = UTIL.flatten(fake_output2);
            float[] fake_output_l = discriminator.dense.forward_Sigmoid(fake_out_l);

            // Calculate Loss
            float gen_loss = UTIL.gen_loss(fake_output_l);
            float disc_loss = UTIL.disc_loss(real_output_l, fake_output_l);

            System.out.println("Gen_Loss " + gen_loss);
            System.out.println("Disc_Loss " + disc_loss);

            // GEN BACKWARD
            System.out.println("Generator Backward");
            float[] fake_gradient_l = UTIL.flatten(fakeImage[0]);
            fake_gradient_l = UTIL.computeGradientFake(fake_gradient_l);
            float[][][] fake_back_gradient = new float[1][28][28];
            fake_back_gradient[0] = UTIL.unflatten(fake_gradient_l, 28, 28);
            float[][][] gradient1 = generator.tconv2.backward(fake_back_gradient, learning_rate_gen);
            float[][][] gradient2 = generator.tconv1.backward(gradient1, learning_rate_gen);
            float[] out = UTIL.flatten(gradient2);
            generator.dense.backward(out, learning_rate_gen);

            // DISC REAL BACKWARD
            System.out.println("Discriminator Backward Real");
            float[] real_gradient_l = UTIL.computeGradientReal(real_out_l);
            real_gradient_l = discriminator.dense.backward(real_gradient_l, learning_rate_disc);
            System.out.printf("Real Gradient Length %d\n", real_gradient_l.length);
            int size = (int) Math.sqrt((float) real_gradient_l.length / discriminator.conv2.filters.length);
            float[][] real_gradient = UTIL.unflatten(real_gradient_l, discriminator.conv2.filters.length, size * size);
            float[][] real_gradient2 = discriminator.conv2.backward(real_gradient, learning_rate_disc);
            discriminator.conv1.backward(real_gradient2, learning_rate_disc);

            // DISC FAKE BACKWARD
            System.out.println("Discriminator Backward Fake");
            float[] fake_gradient_l_1 = UTIL.computeGradientFake(fake_out_l);
            fake_gradient_l = discriminator.dense.backward(fake_gradient_l_1, learning_rate_disc);
            float[][] fake_gradient = UTIL.unflatten(fake_gradient_l, discriminator.conv2.filters.length, size * size);
            float[][] fake_gradient2 = discriminator.conv2.backward(fake_gradient, learning_rate_disc);
            discriminator.conv1.backward(fake_gradient2, learning_rate_disc);

            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    float value = fakeImage[0][y][x];
                    int rgb = (int) ((value + 1) * 255.0f);
                    rgb = (rgb << 16) | (rgb << 8) | rgb;
                    image.setRGB(x, y, rgb);
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
    ConvolutionalLayer conv2;
    DenseLayer dense;

    public Discriminator_Implementation() {
        this.conv1 = new ConvolutionalLayer(5, 64);
        this.conv2 = new ConvolutionalLayer(5, 128);
        this.dense = new DenseLayer(24 * 24 * 128, 1);
    }
}

class Generator_Implementation {
    //output_size = (input_size - 1) * stride + kernel_size - 2 * padding + output_padding + 1
    //output_size = (input_size - 1) * stride + kernel_size + 1 (assuming no padding)
    int dense_output_size;

    DenseLayer dense;
    TransposeConvolutionalLayer tconv1;
    TransposeConvolutionalLayer tconv2;
    TransposeConvolutionalLayer tconv3;

    public Generator_Implementation() {
        this.dense_output_size = 7 * 7 * 128;

        this.dense = new DenseLayer(100, this.dense_output_size);
        this.tconv1 = new TransposeConvolutionalLayer(128, 7, 64, 1);
        this.tconv2 = new TransposeConvolutionalLayer(64, 14, 1, 1);
    }
}

class UTIL {
    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }

    public static float[][] img_to_mat(BufferedImage imageToPixelate) {
        int w = imageToPixelate.getWidth(), h = imageToPixelate.getHeight();
        int[] pixels = imageToPixelate.getRGB(0, 0, w, h, null, 0, w);
        float[][] dta = new float[w][h];

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

    public float gen_loss(float[] fake_output) {
        float[] fake_one = new float[fake_output.length];

        for (int i = 0; i < fake_output.length; i++) {
            fake_one[i] = 1;
        }
        return binary_cross_entropy(fake_one, fake_output);
    }

    public float disc_loss(float[] real_output, float[] fake_output) {
        float[] real_one = new float[real_output.length];
        float[] fake_zero = new float[fake_output.length];

        for (int i = 0; i < real_output.length; i++) {
            real_one[i] = 1;
            fake_zero[i] = 0;
        }

        float real_loss = binary_cross_entropy(real_one, real_output);
        float fake_loss = binary_cross_entropy(fake_zero, fake_output);

        return real_loss + fake_loss;
    }

    public float binary_cross_entropy(float[] y_true, float[] y_pred) {
        float sum = 0.0F;
        float epsilon = 1e-15F; // small value to prevent taking log of zero
        for (int i = 0; i < y_true.length; i++) {
            float pred = (float) Math.max(Math.min(y_pred[i], 1. - epsilon), epsilon); // clamp predictions to avoid log(0)
            sum += (float) (-y_true[i] * Math.log(pred) - (1 - y_true[i]) * Math.log(1 - pred));
        }
        return sum / y_true.length;
    }

    public float[][] unflatten(float[] out_l, int height, int width) {

        float[][] output = new float[height][width];
        int k = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[i][j] = out_l[k++];
            }
        }
        return output;
    }

    public float[] flatten(float[][] input) {
        int totalLength = 0;
        for (float[] arr : input) {
            totalLength += arr.length;
        }
        float[] output = new float[totalLength];
        int k = 0;
        for (float[] floats : input) {
            for (float afloat : floats) {
                output[k++] = afloat;
            }
        }
        return output;
    }

    public float[] flatten(float[][][] input) {
        int actualDepth = input.length;
        int actualHeight = input[0].length;
        int actualWidth = input[0][0].length;
        int m = 0;
        float[] flatten_output = new float[actualDepth * actualHeight * actualWidth];
        for (float[][] floats : input) {
            for (int i = 0; i < actualHeight; i++) {
                for (int j = 0; j < actualWidth; j++) {
                    flatten_output[m++] = floats[i][j];
                }
            }
        }
        return flatten_output;
    }

    public float[] computeGradientReal(float[] real_output) {
        float[] gradient = new float[real_output.length];
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] = 1 / real_output[i];
        }
        return gradient;
    }

    public float[] computeGradientFake(float[] fake_output) {
        float[] gradient = new float[fake_output.length];
        for (int i = 0; i < fake_output.length; i++) {
            gradient[i] = 1 / fake_output[i] - 1;
        }
        return gradient;
    }

//    public float[] computeGradientReal(float[] real_output) {
//        float[] gradient = new float[real_output.length];
//        for (int i = 0; i < real_output.length; i++) {
//            gradient[i] = real_output[i] - 1;
//        }
//        return gradient;
//    }
//
//    public float[] computeGradientFake(float[] fake_output) {
//        float[] gradient = new float[fake_output.length];
//        for (int i = 0; i < fake_output.length; i++) {
//            gradient[i] = fake_output[i];
//        }
//        return gradient;
//    }
}