<<<<<<< HEAD:GAN/src/DCGAN.java
import java.util.ArrayList;
import java.util.List;
=======
<<<<<<< HEAD:misc/GAN/src/DCGAN.java
>>>>>>> 59becac33a818118d87603ccca27a98e925c17e0:misc/GAN/src/DCGAN.java
import java.util.Random;
import java.util.Arrays;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.util.ArrayList;
import java.util.List;
=======
import java.util.*;
import java.awt.image.BufferedImage;
import java.io.*;

import java.io.File;
import javax.imageio.ImageIO;

>>>>>>> 15b5cc1c4f1214daa98dcef0ccf9d8bf8d4875d8:DCGAN.java

public class DCGAN {
    public static final int IMAGE_SIZE = 28;
    public static final int IMAGE_CHANNELS = 1;
    public static final int NOISE_DIM = 100;
    public static final int BATCH_SIZE = 256;
    public static final int BUFFER_SIZE = 60000;
    public static final int EPOCHS = 50;
    public static final int NUM_EXAMPLES_TO_GENERATE = 16;

    private BufferedImage[] trainImages; // Placeholder for training images

    private Generator generator;
    private Discriminator discriminator;
    private Random random = new Random();

    public DCGAN() {
        generator = new Generator(NOISE_DIM, IMAGE_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_CHANNELS, NOISE_DIM);
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
                double[][][] floatImages = convertImagesToFloatArray(batch);
                trainStep(floatImages);
            }

            System.out.println("Time for epoch " + (epoch + 1) + " is "
                    + (System.currentTimeMillis() - startTime) / 1000.0 + " sec");
            generateAndSaveImages(generator, epoch + 1);
        }
    }

    private void trainStep(double[][][] images) {
        double[][] noise = new double[BATCH_SIZE][NOISE_DIM];
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < NOISE_DIM; j++) {
                noise[i][j] = random.nextGaussian();
            }
        }

        // Forward pass
        double[][] generatedImages = generator.forward(noise);
        double[] realOutput = discriminator.forward(images);

        // Divide generated images into batches of 8
        int batchSize = 8;
        int numBatches = generatedImages.length / batchSize;

        // Create 3D array to store batches
        double[][][] images_array = new double[batchSize][IMAGE_SIZE][IMAGE_SIZE];

        // Populate the imageBatches array
        for (int image_idx = 0; image_idx < batchSize; image_idx++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                for (int k = 0; k < IMAGE_SIZE; k++) {
                    images_array[image_idx][j][k] = generatedImages[image_idx][j * IMAGE_SIZE + k];
                }
            }
        }
        double[] fakeOutput = discriminator.forward(images_array);

        // Compute loss
        double genLoss = generatorLoss(fakeOutput);
        double discLoss = discriminatorLoss(realOutput, fakeOutput);

        // Backward pass
        double[][] genGradients = generator.backward(genLoss, noise);
        double[][] discGradients = discriminator.backward(discLoss, images, generatedImages, null, null);

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

    private double[][][] convertImagesToFloatArray(BufferedImage[] images) {
        int batchSize = images.length;
        double[][][] floatImages = new double[batchSize][IMAGE_SIZE][IMAGE_SIZE];
        for (int i = 0; i < batchSize; i++) {
            float[][] imageArray = img_to_mat(images[i]);
            for (int x = 0; x < IMAGE_SIZE; x++) {
                for (int y = 0; y < IMAGE_SIZE; y++) {
                    floatImages[i][x][y] = imageArray[x][y];
                }
            }
        }
        return floatImages;
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

    private float[][] img_to_mat(BufferedImage image) {
        // Convert BufferedImage to float array
        int width = image.getWidth();
        int height = image.getHeight();
        float[][] mat = new float[width][height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                mat[x][y] = (float) (image.getRGB(x, y) & 0xFF) / 255.0f;
            }
        }
        return mat;
    }
}

class Generator {
    private double[][] weights;
    private double[] biases;
    private final int IMAGE_SIZE, IMAGE_CHANNELS, NOISE_DIM;

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
                                    // images = [batchSize][imageSize * imageSize * imageChannels]
                                    sum += images[b][i + fi][j + fj][c] * convWeights[fi][fj][c][f];
                                }
                            }
                        }
<<<<<<< HEAD:GAN/src/DCGAN.java
                        convOutput[b * convOutputSize
                                + f * ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2)
                                + (i / 2) * ((imageSize - filterSize + 1) / 2) + (j / 2)] = Math
                                        .max(0.01f * (sum + convBiases[f]), sum + convBiases[f]);
=======
                        convOutput[b * convOutputSize + f * ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2) + (i / 2) * ((imageSize - filterSize + 1) / 2) + (j / 2)] = Math.max(0.01 * (sum + convBiases[f]), sum + convBiases[f]);
>>>>>>> 59becac33a818118d87603ccca27a98e925c17e0:misc/GAN/src/DCGAN.java
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

<<<<<<< HEAD:GAN/src/DCGAN.java
    public double[][] backward(double loss, double[][][] realImages, double[][] generatedImages, double[][] convOutput,
            double[][][][] images) {
=======
    public double[][][] backward(double loss, double[][][] realImages, double[][] generatedImages, double[][] convOutput, double[][][][] images) {
>>>>>>> 59becac33a818118d87603ccca27a98e925c17e0:misc/GAN/src/DCGAN.java
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
        for (int b = 0; b < batchSize; b++) {
<<<<<<< HEAD:GAN/src/DCGAN.java
            double dOut = loss * (realImages[b][0][0] - generatedImages[b][0]); // Assuming loss is binary cross-entropy
                                                                                // for single output
=======
            double dOut = loss * (realImages[b][0][0] - generatedImages[b][0]); // Assuming loss is binary cross-entropy for single output
>>>>>>> 59becac33a818118d87603ccca27a98e925c17e0:misc/GAN/src/DCGAN.java
            dDenseBias += dOut;
            for (int i = 0; i < convOutputSize; i++) {
                dDenseWeights[i] += dOut * convOutput[b][i] * (1 - convOutput[b][i]); // Update with sigmoid derivative
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
<<<<<<< HEAD:GAN/src/DCGAN.java
                            // new double[filterSize][filterSize][imageChannels][numFilters];
                            dConvWeights[fi][fj][c][f] += dSum * images[b][i + fi][j + fj][c];
=======
                            for (int fi = 0; fi < filterSize; fi++) {
                                for (int fj = 0; fj < filterSize; fj++) {
                                    dConvWeights[fi][fj][c][f] += dSum * images[b][i + fi][j + fj][c];
                                }
                            }
>>>>>>> 59becac33a818118d87603ccca27a98e925c17e0:misc/GAN/src/DCGAN.java
                        }
                        dConvBiases[f] += dSum;
                    }
                }
            }
        }

        // This function returns the gradients.
        // You can modify it to return what's needed.
<<<<<<< HEAD:GAN/src/DCGAN.java
        // For example, you can return a list containing {dConvWeights, dConvBiases,
        // dDenseWeights, dDenseBias}
        return new double[][][] { dConvWeights, new double[][] { dConvBiases }, new double[][] { dDenseWeights },
                new double[][] { { dDenseBias } } };
=======
        // For example, you can return a list containing {dConvWeights, dConvBiases, dDenseWeights, dDenseBias}
        return new double[][][]{dConvWeights, dConvBiases, new double[][]{dDenseWeights}, new double[][]{{dDenseBias}}};
>>>>>>> 59becac33a818118d87603ccca27a98e925c17e0:misc/GAN/src/DCGAN.java
    }

    // public void updateParameters(double[][] gradients) {
    // // Update parameters implementation
    // // Not implemented for simplicity (can be added if needed)
    // }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
<<<<<<< HEAD:misc/GAN/src/DCGAN.java
}
<<<<<<< HEAD:GAN/src/DCGAN.java
=======
=======
}
>>>>>>> 15b5cc1c4f1214daa98dcef0ccf9d8bf8d4875d8:DCGAN.java
>>>>>>> 59becac33a818118d87603ccca27a98e925c17e0:misc/GAN/src/DCGAN.java
