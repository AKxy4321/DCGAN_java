import java.util.*;
import java.awt.image.BufferedImage;
import java.io.*;

import java.io.File;
import javax.imageio.ImageIO;


public class DCGAN {
    private static final int IMAGE_SIZE = 28;
    private static final int IMAGE_CHANNELS = 1;
    private static final int NOISE_DIM = 100;
    private static final int BATCH_SIZE = 256;
    private static final int BUFFER_SIZE = 60000;
    private static final int EPOCHS = 50;
    private static final int NUM_EXAMPLES_TO_GENERATE = 16;

    private double[][][] trainImages; // Placeholder for training images

    private Generator generator;
    private Discriminator discriminator;
    private Random random = new Random();

    public DCGAN() {
        generator = new Generator(NOISE_DIM, IMAGE_SIZE, IMAGE_CHANNELS);
        discriminator = new Discriminator(IMAGE_SIZE, IMAGE_CHANNELS);
    }   

    public void train() {
        int label_counter = 0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            long startTime = System.currentTimeMillis();

            List<BufferedImage> realImages = new ArrayList<>();
            for (int i = 0; i < BUFFER_SIZE; i++) {
                BufferedImage image = mnist_load_random(label_counter);
                realImages.add(image);
            }
            if(label_counter==9){
                label_counter=0;
            }else{
                label_counter++;
            }

            double[][][] realImageBatch = convertImagesToFloatArray(realImages);

            for (int i = 0; i < trainImages.length; i += BATCH_SIZE) {
                double[][][] batch = getBatch(trainImages, i, BATCH_SIZE);
                trainStep(batch);
            }

            System.out.println("Time for epoch " + (epoch + 1) + " is " + (System.currentTimeMillis() - startTime) / 1000.0 + " sec");
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
        double[] fakeOutput = discriminator.forward(generatedImages);

        // Compute loss
        double genLoss = generatorLoss(fakeOutput);
        double discLoss = discriminatorLoss(realOutput, fakeOutput);

        // Backward pass
        double[][] genGradients = generator.backward(genLoss, noise);
        double[][] discGradients = discriminator.backward(discLoss, images, generatedImages);

        // Update parameters
        generator.updateParameters(genGradients);
        discriminator.updateParameters(discGradients);
    }

    public static BufferedImage mnist_load_random(int label) throws IOException {
        String mnist_path = "data\\mnist_png\\mnist_png\\testing";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        assert files != null;
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "\\" + label + "\\" + files[random_index];
        return load_image(final_path);
    }

    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }

    private double[][][] convertImagesToFloatArray(List<BufferedImage> images) {
        int batchSize = images.size();
        double[][][] floatImages = new double[batchSize][IMAGE_SIZE][IMAGE_SIZE];
        for (int i = 0; i < batchSize; i++) {
            BufferedImage image = images.get(i);
            float[][] imageArray = img_to_mat(image);
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

    private double[][][] getBatch(double[][][] data, int start, int batchSize) {
        double[][][] batch = new double[batchSize][IMAGE_SIZE][IMAGE_SIZE];
        System.arraycopy(data, start, batch, 0, batchSize);
        return batch;
    }

    public static void main(String[] args) {
        DCGAN dcgan = new DCGAN();
        dcgan.train();
    }
}

class Generator {
    private double[][] weights;
    private double[] biases;

    public Generator(int noiseDim, int imageSize, int imageChannels) {
        // Initialize weights and biases randomly
        weights = new double[noiseDim][imageSize * imageSize * imageChannels];
        biases = new double[imageSize * imageSize * imageChannels];
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
        int imageSize = IMAGE_SIZE;
        int imageChannels = IMAGE_CHANNELS;
        double[][] output = new double[batchSize][imageSize * imageSize * imageChannels];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < imageSize * imageSize * imageChannels; j++) {
                double sum = biases[j];
                for (int k = 0; k < NOISE_DIM; k++) {
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
        int imageSize = IMAGE_SIZE;
        int imageChannels = IMAGE_CHANNELS;
        double[][] gradients = new double[NOISE_DIM][imageSize * imageSize * imageChannels];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < imageSize * imageSize * imageChannels; j++) {
                for (int k = 0; k < NOISE_DIM; k++) {
                    gradients[k][j] += loss * weights[k][j];
                }
            }
        }

        return gradients;
    }

    public void updateParameters(double[][] gradients) {
        // Update parameters implementation
        int imageSize = IMAGE_SIZE;
        int imageChannels = IMAGE_CHANNELS;
        double learningRate = 0.001; // Adjust as needed

        for (int k = 0; k < NOISE_DIM; k++) {
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
                                    sum += images[b][i + fi][j + fj][c] * convWeights[fi][fj][c][f];
                                }
                            }
                        }
                        convOutput[b * convOutputSize + f * ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2) + (i / 2) * ((imageSize - filterSize + 1) / 2) + (j / 2)] = Math.max(0.01f * (sum + convBiases[f]), sum + convBiases[f]);
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

    public double[][][] backward(double loss, double[][][] realImages, double[][] generatedImages, double[][] convOutput, double[][][][] images) {
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
            double dOut = loss * (realImages[b][0][0] - generatedImages[b][0]); // Assuming loss is binary cross-entropy for single output
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
                        double dSum = dConvOutput[b][f * ((imageSize - filterSize + 1) / 2) * ((imageSize - filterSize + 1) / 2) + (i / 2) * ((imageSize - filterSize + 1) / 2) + (j / 2)][0];
                        for (int c = 0; c < imageChannels; c++) {
                                    dConvWeights[fi][fj][c][f] += dSum * images[b][i + fi][j + fj][c];
                                }
                            dConvBiases[f] += dSum;
                        }
                    }   
                }
            }

        // This function returns the gradients.
        // You can modify it to return what's needed.
        // For example, you can return a list containing {dConvWeights, dConvBiases, dDenseWeights, dDenseBias}
        return new double[][][]{dConvWeights, new double[][]{dConvBiases}, new double[][]{dDenseWeights}, new double[][]{{dDenseBias}}};
    }

    // public void updateParameters(double[][] gradients) {
    //     // Update parameters implementation
    //     // Not implemented for simplicity (can be added if needed)
    // }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}