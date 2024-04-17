import java.util.ArrayList;
import java.util.List;

import java.util.Random;
import java.util.Arrays;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.util.ArrayList;
import java.util.List;
import java.util.*;
import java.awt.image.BufferedImage;
import java.io.*;

import java.io.File;
import javax.imageio.ImageIO;

public class DCGAN {
    public static final int IMAGE_SIZE = 28;
    public static final int IMAGE_CHANNELS = 1;
    public static final int NOISE_DIM = 100;
    public static final int BATCH_SIZE = 256;
    public static final int BUFFER_SIZE = 60000;
    public static final int EPOCHS = 50;
    public static final int NUM_EXAMPLES_TO_GENERATE = 16;

    private BufferedImage[] trainImages; // Placeholder for training images

    private GeneratorModel generator;
    private Discriminator discriminator;
    private Random random = new Random();

    public DCGAN() {
        // TODO: Make it customizable instead of hardcoding later
        generator = new GeneratorModel(); // NOISE_DIM, IMAGE_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_CHANNELS,
                                          // NOISE_DIM);
        discriminator = new Discriminator(IMAGE_SIZE, IMAGE_CHANNELS);
    }

    public void train() throws IOException {
        int label_counter = 0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            long startTime = System.currentTimeMillis();

            List<BufferedImage> realImages = new ArrayList<>();
            for (int i = 0; i < BUFFER_SIZE; i++) {
                BufferedImage image = mnist_load_random(label_counter);
                realImages.add(image);
            }
            if (label_counter == 9) {
                label_counter = 0;
            } else {
                label_counter++;
            }

            trainImages = realImages.toArray(new BufferedImage[0]);

            for (int i = 0; i < trainImages.length; i += BATCH_SIZE) {
                BufferedImage[] batch = getBatch(trainImages, i, BATCH_SIZE);
                double[][][] doubleImages = convertImagesTodoubleArray(batch);// idx,i,j
                trainStep(doubleImages);
            }

            System.out.println("Time for epoch " + (epoch + 1) + " is "
                    + (System.currentTimeMillis() - startTime) / 1000.0 + " sec");
            generateAndSaveImages(generator, epoch + 1);
        }
    }

    private void trainStep(double[][][] images) {
        double[][] noises = new double[BATCH_SIZE][NOISE_DIM];
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < NOISE_DIM; j++) {
                noises[i][j] = random.nextGaussian();
            }
        }

        // ok, so for each image, we have only a 2d array accessed by y,x indices,
        // but the discriminator expects it to have an image channel index also
        // , so we have to add an extra index for the channel (c)
        // We also have to store the each image in an array, so one more index to acces
        // to the image (image_idx)

        double[][] generatedImages = generator.forward(noises);

        double[][][][] modified_generated_images_array = new double[BATCH_SIZE][IMAGE_SIZE][IMAGE_SIZE][IMAGE_CHANNELS];
        convert_generated_imgs_to_format: {
            for (int image_idx = 0; image_idx < BATCH_SIZE; image_idx++) {
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    for (int k = 0; k < IMAGE_SIZE; k++) {
                        for (int c = 0; c < IMAGE_CHANNELS; c++)
                            modified_generated_images_array[image_idx][j][k][c] = generatedImages[image_idx][j
                                    * IMAGE_SIZE + k];
                    }
                }
            }
        }

        double[][][][] modified_real_images_array = new double[BATCH_SIZE][IMAGE_SIZE][IMAGE_SIZE][IMAGE_CHANNELS];
        convert_real_imgs_to_format: {
            for (int image_idx = 0; image_idx < BATCH_SIZE; image_idx++) {
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    for (int k = 0; k < IMAGE_SIZE; k++) {
                        for (int c = 0; c < IMAGE_CHANNELS; c++)
                            modified_real_images_array[image_idx][j][k][c] = images[image_idx][j][k];
                    }
                }
            }
        }

        // Forward pass
        double[] realOutput = discriminator.forward(modified_real_images_array);

        double[] fakeOutput = discriminator.forward(modified_generated_images_array);

        // Compute loss
        double genLoss = generatorLoss(fakeOutput);
        double discLoss = discriminatorLoss(realOutput, fakeOutput);

        // TODO : rewrite all 4 methods being used below

        // Backward pass
        double[][] genGradients = generator.backward(genLoss, noises);

        // TODO: rewrite backward propagation code of discriminator class because idk how convOutput is supposed to be made 
        double[][] discGradients = discriminator.backward(discLoss, modified_real_images_array, modified_generated_images_array, null);

        // Update parameters
        generator.updateParameters(genGradients);
        discriminator.updateParameters(discGradients);
    }

    public static BufferedImage mnist_load_random(int label) throws IOException {
        String mnist_path = "data/mnist_png/mnist_png/testing";
        File dir = new File(mnist_path + "/" + label);
        String[] files = dir.list();
        assert files != null;
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "/" + label + "/" + files[random_index];
        return load_image(final_path);
    }

    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }

    private double[][][] convertImagesTodoubleArray(BufferedImage[] images) {
        int batchSize = images.length;
        double[][][] doubleImages = new double[batchSize][IMAGE_SIZE][IMAGE_SIZE];
        for (int i = 0; i < batchSize; i++) {
            double[][] imageArray = img_to_mat(images[i]);
            for (int x = 0; x < IMAGE_SIZE; x++) {
                for (int y = 0; y < IMAGE_SIZE; y++) {
                    doubleImages[i][x][y] = imageArray[x][y];
                }
            }
        }
        return doubleImages;
    }

    private double generatorLoss(double[] fakeOutput) {
        double loss = 0.0;
        for (double output : fakeOutput) {
            loss += Math.log(1.0 - output); // Binary cross-entropy loss
        }
        return -loss / fakeOutput.length;
    }

    private double discriminatorLoss(double[] realOutput, double[] fakeOutput) {
        double realLoss = 0.0;
        for (double output : realOutput) {
            realLoss += Math.log(output); // Binary cross-entropy loss
        }
        double fakeLoss = 0.0;
        for (double output : fakeOutput) {
            fakeLoss += Math.log(1.0 - output); // Binary cross-entropy loss
        }
        return -(realLoss / realOutput.length + fakeLoss / fakeOutput.length);
    }

    private void generateAndSaveImages(Generator generator, int epoch) {
        double[][] noise = new double[NUM_EXAMPLES_TO_GENERATE][NOISE_DIM];
        for (int i = 0; i < NUM_EXAMPLES_TO_GENERATE; i++) {
            for (int j = 0; j < NOISE_DIM; j++) {
                noise[i][j] = random.nextGaussian();
            }
        }

        double[][] generatedImages = generator.forward(noise);
        // Save generated images
        System.out.println("Generated images at epoch " + epoch + ":");
        for (int i = 0; i < NUM_EXAMPLES_TO_GENERATE; i++) {
            System.out.println(Arrays.toString(generatedImages[i]));
        }
    }

    private BufferedImage[] getBatch(BufferedImage[] data, int start, int batchSize) {
        BufferedImage[] batch = new BufferedImage[batchSize];
        System.arraycopy(data, start, batch, 0, batchSize);
        return batch;
    }

    public static void main(String[] args) {
        DCGAN dcgan = new DCGAN();
        try {
            dcgan.train();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private double[][] img_to_mat(BufferedImage image) {
        // Convert BufferedImage to double array
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] mat = new double[width][height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                mat[x][y] = (double) (image.getRGB(x, y) & 0xFF) / 255.0f;
            }
        }
        return mat;
    }
}

class GeneratorModel {

    private static final int INPUT_SIZE = 100;
    private static final int HIDDEN_DIM = 7 * 7 * 256;
    private static final int FILTER1_SIZE = 5;
    private static final int IMAGE_SIZE = 28;
    private static final int IMAGE_CHANNELS = 1;

    // rivate static final int FILTER2_SIZE = 5;

    private final double[] weights1; // Dense layer weights (input -> hidden)
    private final double[] weights2; // Conv2DTranspose layer weights (hidden -> 7x7x128)
    private final double[] weights3; // Conv2DTranspose layer weights (128 -> 14x14x64)
    private final double[] weights4; // Conv2DTranspose layer weights (64 -> 28x28x1)
    private final double learningRate = 0.001f;

    public GeneratorModel() {
        weights1 = initializeWeights(INPUT_SIZE * HIDDEN_DIM);
        weights2 = initializeWeights(HIDDEN_DIM * 7 * 7 * 128);
        weights3 = initializeWeights(128 * 14 * 14 * 64);
        weights4 = initializeWeights(64 * 28 * 28);
    }

    // public double computeGeneratorLoss(double[] input) {
    // double[] fakeImage = generate(input); // Generate fake image
    // double[] fakeOutput = getDiscriminatorOutput(fakeImage); // Get
    // discriminator's output for fake image
    // return binaryCrossEntropyLoss(fakeOutput); // Compute binary cross-entropy
    // loss
    // }

    private double binaryCrossEntropyLoss(double[] predictions) {
        // Assuming predictions contain the discriminator's output for fake images
        double loss = 0.0f;
        for (double prediction : predictions) {
            // Assuming the expected label for fake images is 1 (indicating real)
            // You may need to adjust this based on your specific implementation
            loss += (double) -Math.log(prediction); // Compute negative log likelihood
        }
        // Normalize the loss by the number of predictions
        return loss / predictions.length;
    }

    // public double[][] backward(double loss, double[][] noises) {
    //     // Backward pass implementation
    //     // int batchSize = noises.length;
    //     // return gradients;
    // }

    private void updateParameters(double[] input, double[] gradients) {
        for (int i = 0; i < weights1.length; i++) {
            weights1[i] -= learningRate * gradients[i] * input[i % input.length]; // Update weights1
        }
        for (int i = 0; i < weights2.length; i++) {
            weights2[i] -= learningRate * gradients[i] * input[i % input.length]; // Update weights2
        }
        for (int i = 0; i < weights3.length; i++) {
            weights3[i] -= learningRate * gradients[i] * input[i % input.length]; // Update weights3
        }
        for (int i = 0; i < weights4.length; i++) {
            weights4[i] -= learningRate * gradients[i] * input[i % input.length]; // Update weights4
        }
    }

    public double[][] forward(double[] input) {// used to generate image also
        double[] hidden = dense(input);
        double[] conv1 = conv2dTranspose(hidden, weights2, 7, 7, 128);
        double[] conv2 = conv2dTranspose(conv1, weights3, 14, 14, 64);
        double[] final_layer_output = conv2dTranspose(conv2, weights4, 28, 28, 1);

        double[][] output = new double[28][28];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int index = (y * 28) + x;
                output[y][x] = final_layer_output[index];
            }
        }
        return output;
    }

    public double[][] forward(double[][] input) {// used to generate image also
        int BATCH_SIZE = input.length, output_width = 28, output_height = 28;

        double[][] output = new double[BATCH_SIZE][output_width * output_height];
        for (int i = 0; i < BATCH_SIZE; i++) {
            double unflattened_output[][] = forward(input[i]);

            // flattening the x,y indices to just one indexes
            for (int j = 0; j < output_width; j++) {
                for (int k = 0; k < output_height; k++) {
                    output[i][j * output_width + k] = unflattened_output[j][k];
                }
            }
        }

        return output;
    }

    private double[] dense(double[] input) {
        int numInputs = input.length;
        double[] hidden = new double[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            double sum = 0.0f;
            for (int j = 0; j < numInputs; j++) {
                sum += input[j] * weights1[i * numInputs + j];
            }
            hidden[i] = leakyReLU(sum); // Apply LeakyReLU activation
        }
        return hidden;
    }

    private double leakyReLU(double x) {
        return Math.max(0.01f * x, x); // LeakyReLU with slope 0.01
    }

    private double[] conv2dTranspose(double[] input, double[] weights, int outputWidth, int outputHeight,
            int numFilters) {
        int numInputs = input.length / (outputWidth * outputHeight);
        double[] output = new double[outputWidth * outputHeight * numFilters];
        for (int filter = 0; filter < numFilters; filter++) {
            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    double sum = 0.0f;
                    for (int inY = 0; inY < FILTER1_SIZE; inY++) {
                        for (int inX = 0; inX < FILTER1_SIZE; inX++) {
                            int inIndex = (y * outputWidth + x) * numInputs + inY * outputWidth + inX;
                            if (inY + y < outputHeight && inX + x < outputWidth) { // Handle padding (same)
                                sum += input[inIndex] * weights[filter * numInputs * FILTER1_SIZE * FILTER1_SIZE
                                        + inY * FILTER1_SIZE + inX];
                            }
                        }
                    }
                    output[(filter * outputHeight + y) * outputWidth + x] = leakyReLU(sum); // Apply LeakyReLU
                                                                                            // activation
                }
            }
        }
        return output;
    }

    public static void main(String[] args) {
        // Example weights (replace with actual weights from training)
        // double[] weights1 = initializeWeights(INPUT_SIZE * HIDDEN_DIM);
        // double[] weights2 = initializeWeights(HIDDEN_DIM * 7 * 7 * 128);
        // double[] weights3 = initializeWeights(128 * 14 * 14 * 64);
        // double[] weights4 = initializeWeights(64 * 28 * 28);

        GeneratorModel model = new GeneratorModel();// weights1, weights2, weights3, weights4);

        // Generate random input noise
        double[] input = generateGaussianNoise();

        double[][] output = model.forward(input);

        // Display some pixel values
        for (int i = 0; i < 10; i++) {
            System.out.println("Pixel value at index " + i + ",0: " + output[i][0]);
        }

        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                double value = output[y][x]; // Assuming all channels have the same value
                int rgb = (int) ((value + 1) * 255.0f); // Scale to 0-255 range
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

    private static double[] initializeWeights(int size) {
        double[] weights = new double[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            weights[i] = (double) random.nextGaussian() * 0.14f; // Initialize with random values from normal
                                                                 // distribution
        }
        return weights;
    }

    public static double[] generateGaussianNoise() {
        double[] noise = new double[GeneratorModel.INPUT_SIZE];
        Random random = new Random();
        for (int i = 0; i < GeneratorModel.INPUT_SIZE; i++) {
            noise[i] = (double) random.nextGaussian(); // Generate noise from normal distribution (between -1 and 1)
        }
        return noise;
    }
}

class Generator {
    private double[][] weights;
    private double[] biases;
    private final int IMAGE_SIZE, IMAGE_CHANNELS, NOISE_DIM;
    private static final int HIDDEN_DIM = 7 * 7 * 256;
    private static final int FILTER1_SIZE = 5;

    public Generator(int noiseDim, int imageSize, int imageChannels, int IMAGE_SIZE, int IMAGE_CHANNELS,
            int NOISE_DIM) {
        // Initialize weights and biases randomly
        weights = new double[noiseDim][imageSize * imageSize * imageChannels];
        biases = new double[imageSize * imageSize * imageChannels];

        this.IMAGE_SIZE = IMAGE_SIZE;
        this.IMAGE_CHANNELS = IMAGE_CHANNELS;
        this.NOISE_DIM = NOISE_DIM;

        // Initialize weights and biases using Xavier initialization
        double scale = Math.sqrt(2.0 / (noiseDim + imageSize * imageSize * imageChannels));
        Random random = new Random();
        for (int i = 0; i < noiseDim; i++) {
            for (int j = 0; j < imageSize * imageSize * imageChannels; j++) {
                weights[i][j] = random.nextGaussian() * scale;
            }
        }
        for (int i = 0; i < imageSize * imageSize * imageChannels; i++) {
            biases[i] = random.nextGaussian() * scale;
        }
    }

    public double[][] forward(double[][] noise) {

        // Forward pass implementation
        int batchSize = noise.length;
        int imageSize = DCGAN.IMAGE_SIZE;
        int imageChannels = DCGAN.IMAGE_CHANNELS;
        double[][] output = new double[batchSize][imageSize * imageSize * imageChannels];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < imageSize * imageSize * imageChannels; j++) {
                double sum = biases[j];
                for (int k = 0; k < DCGAN.NOISE_DIM; k++) {
                    sum += noise[i][k] * weights[k][j];
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    public double[][] backward(double loss, double[][] noise) {
        // Backward pass implementation
        // Compute gradients w.r.t. weights and biases
        int batchSize = noise.length;
        int imageSize = DCGAN.IMAGE_SIZE;
        int imageChannels = DCGAN.IMAGE_CHANNELS;
        double[][] gradients = new double[DCGAN.NOISE_DIM][imageSize * imageSize * imageChannels];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < imageSize * imageSize * imageChannels; j++) {
                for (int k = 0; k < DCGAN.NOISE_DIM; k++) {
                    gradients[k][j] += loss * weights[k][j];
                }
            }
        }

        return gradients;
    }

    public void updateParameters(double[][] gradients) {
        // Update parameters implementation
        int imageSize = DCGAN.IMAGE_SIZE;
        int imageChannels = DCGAN.IMAGE_CHANNELS;
        double learningRate = 0.001; // Adjust as needed

        for (int k = 0; k < DCGAN.NOISE_DIM; k++) {
            for (int j = 0; j < imageSize * imageSize * imageChannels; j++) {
                weights[k][j] -= learningRate * gradients[k][j];
            }
        }

        // Update biases
        for (int j = 0; j < imageSize * imageSize * imageChannels; j++) {
            biases[j] -= learningRate * gradients[0][j];
        }
    }
}

class Discriminator {
    private double[][][][] convWeights;
    private double[] convBiases;
    private double[] denseWeights;
    private double denseBias;

    public Discriminator(int imageSize, int imageChannels) {
        // Initialize convolutional layer weights and biases randomly
        int filterSize = 5; // Filter size for convolutional layers
        int numFilters = 64; // Number of filters for each convolutional layer
        convWeights = new double[filterSize][filterSize][imageChannels][numFilters];
        convBiases = new double[numFilters];
        initializeConvWeights();

        // Initialize dense layer weights and bias randomly
        int convOutputSize = ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2) * numFilters;
        denseWeights = new double[convOutputSize];
        denseBias = 0;
        initializeDenseWeights();
    }

    private void initializeConvWeights() {
        Random random = new Random();
        for (int i = 0; i < convWeights.length; i++) {
            for (int j = 0; j < convWeights[0].length; j++) {
                for (int k = 0; k < convWeights[0][0].length; k++) {
                    for (int l = 0; l < convWeights[0][0][0].length; l++) {
                        convWeights[i][j][k][l] = random.nextGaussian() * 0.02; // Initialize with small random values
                    }
                }
            }
        }
        for (int i = 0; i < convBiases.length; i++) {
            convBiases[i] = 0;
        }
    }

    private void initializeDenseWeights() {
        Random random = new Random();
        for (int i = 0; i < denseWeights.length; i++) {
            denseWeights[i] = random.nextGaussian() * 0.02; // Initialize with small random values
        }
    }

    public double[] forward(double[][][][] images) {
        // Forward pass implementation
        int batchSize = images.length;
        int imageSize = images[0].length;
        int imageChannels = images[0][0].length;
        int filterSize = 5; // Filter size for convolutional layers
        int numFilters = 64; // Number of filters for each convolutional layer
        int convOutputSize = ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2) * numFilters;
        double[] convOutput = new double[batchSize * convOutputSize];

        // Convolutional layers
        for (int b = 0; b < batchSize; b++) {
            for (int f = 0; f < numFilters; f++) {
                for (int i = 0; i < imageSize - filterSize + 1; i += 2) {
                    for (int j = 0; j < imageSize - filterSize + 1; j += 2) {
                        double sum = 0;
                        for (int c = 0; c < imageChannels; c++) {
                            for (int fi = 0; fi < filterSize; fi++) {
                                for (int fj = 0; fj < filterSize; fj++) {
                                    // idx,y,x,channel
                                    sum += images[b][i + fi][j + fj][c] * convWeights[fi][fj][c][f];
                                }
                            }
                        }
                        convOutput[b * convOutputSize
                                + f * ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2)
                                + (i / 2) * ((imageSize - filterSize + 1) / 2) + (j / 2)] = Math
                                        .max(0.01f * (sum + convBiases[f]), sum + convBiases[f]);

                    }
                }
            }
        }

        // Dense layer
        double[] output = new double[batchSize];
        for (int b = 0; b < batchSize; b++) {
            double sum = 0;
            for (int i = 0; i < convOutputSize; i++) {
                sum += convOutput[b * convOutputSize + i] * denseWeights[i];
            }
            output[b] = sigmoid(sum + denseBias);
        }

        return output;
    }

    public Object[] backward(double loss, double[][][][] realImages,  double[][][][] generatedImages, double[][] convOutput) {
        int batchSize = realImages.length;
        int imageSize = realImages[0].length;
        int imageChannels = realImages[0][0].length;
        int filterSize = 5; // Filter size for convolutional layers
        int numFilters = 64; // Number of filters for each convolutional layer
        int convOutputSize = ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2) * numFilters;

        // Initialize gradients for weights and biases
        double[][][][] dConvWeights = new double[filterSize][filterSize][imageChannels][numFilters];
        double[] dConvBiases = new double[numFilters];
        double[] dDenseWeights = new double[convOutputSize];
        double dDenseBias = 0;

        // Backpropagate through dense layer
        for (int img_idx = 0; img_idx < batchSize; img_idx++) {
            double dOut = loss * (realImages[img_idx][0][0][0] - generatedImages[img_idx][0][0][0]); // Assuming loss is binary cross-entropy
                                                                                // for single output
            dDenseBias += dOut;
            for (int i = 0; i < convOutputSize; i++) {
                dDenseWeights[i] += dOut * convOutput[img_idx][i] * (1 - convOutput[img_idx][i]); // Update with sigmoid derivative
            }
        }

        // Backpropagate through convolutional layers
        double[][][] dConvOutput = new double[batchSize][convOutputSize][1]; // Assuming single channel output
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < convOutputSize; i++) {
                dConvOutput[b][i][0] = dDenseWeights[i] * convOutput[b][i] * (1 - convOutput[b][i]);
            }
        }

        // Update gradients for convolutional weights and biases
        for (int b = 0; b < batchSize; b++) {
            for (int f = 0; f < numFilters; f++) {
                for (int i = 0; i < imageSize - filterSize + 1; i += 2) {
                    for (int j = 0; j < imageSize - filterSize + 1; j += 2) {
                        double dSum = dConvOutput[b][f * ((imageSize - filterSize + 1) / 2)
                                * ((imageSize - filterSize + 1) / 2) + (i / 2) * ((imageSize - filterSize + 1) / 2)
                                + (j / 2)][0];
                        for (int c = 0; c < imageChannels; c++) {
                            for (int fi = 0; fi < filterSize; fi++) {
                                for (int fj = 0; fj < filterSize; fj++) {
                                    dConvWeights[fi][fj][c][f] += dSum * realImages[b][i + fi][j + fj][c];
                                }
                            }
                        }
                        dConvBiases[f] += dSum;
                    }
                }
            }
        }

        // This function returns the gradients.
        // You can modify it to return what's needed.
        // For example, you can return a list containing {dConvWeights, dConvBiases,
        // dDenseWeights, dDenseBias}
        return new Object[] { dConvWeights, new double[][] { dConvBiases }, new double[][] { dDenseWeights },
                new double[][] { { dDenseBias } } };
        // For example, you can return a list containing {dConvWeights, dConvBiases,
        // dDenseWeights, dDenseBias}
    }

    // public void updateParameters(double[][] gradients) {
    //     // Update parameters implementation
    //     // Not implemented for simplicity (can be added if needed

    //     // Update parameters implementation
    //     int imageSize = DCGAN.IMAGE_SIZE;
    //     int imageChannels = DCGAN.IMAGE_CHANNELS;
    //     double learningRate = 0.001; // Adjust as needed

    //     // TODO: write logic to update weights and biases using gradients
    // }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

}