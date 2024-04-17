import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.util.Random;

public class Generator {
    // Constants
    private static final int IMAGE_SIZE = 28; // Size of the generated image (28x28 pixels for MNIST)
    private static final int NOISE_SIZE = 100; // Size of the input noise vector

    // Generator network parameters
    private double[][] genWeights1;
    private double[][] genWeights2;
    private double[] genBias1;
    private double[] genBias2;

    // Random number generator
    private Random random;

    public Generator() {
        // Initialize random number generator
        random = new Random();

        // Initialize generator network parameters randomly
        genWeights1 = randomWeights(NOISE_SIZE, 100);
        genBias1 = randomBiases(100);
        genWeights2 = randomWeights(100, IMAGE_SIZE * IMAGE_SIZE);
        genBias2 = randomBiases(IMAGE_SIZE * IMAGE_SIZE);
    }

    // Generate random weights for a layer
    private double[][] randomWeights(int inputSize, int outputSize) {
        double[][] weights = new double[inputSize][outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextGaussian();
            }
        }
        return weights;
    }

    // Generate random biases for a layer
    private double[] randomBiases(int size) {
        double[] biases = new double[size];
        for (int i = 0; i < size; i++) {
            biases[i] = random.nextGaussian();
        }
        return biases;
    }

    // Generator network forward pass
    public double[] generateImage(double[] noise) {
        double[] hiddenLayer = MatrixUtils.sigmoid(MatrixUtils.add(MatrixUtils.dot(noise, genWeights1), genBias1));
        return MatrixUtils.sigmoid(MatrixUtils.add(MatrixUtils.dot(hiddenLayer, genWeights2), genBias2));
    }

    // Save the generated image
    private void saveImage(double[] image, String outputDir, int index) {
        BufferedImage bufferedImage = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int grayValue = (int) (image[y + x] * 255.0); // Convert normalized value back to grayscale
                grayValue = Math.min(255, Math.max(0, grayValue)); // Ensure grayValue is within [0, 255]
                int rgb = (grayValue << 16) | (grayValue << 8) | grayValue; // Grayscale to RGB
                bufferedImage.setRGB(x, y, rgb);
            }
        }

        // Write the image to a file
        File outputFile = new File(outputDir, "generated_image_" + index + ".png");
        try {
            ImageIO.write(bufferedImage, "png", outputFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Generate random noise
    private double[] generateNoise(int size) {
        double[] noise = new double[size];
        for (int i = 0; i < size; i++) {
            noise[i] = random.nextGaussian();
        }
        return noise;
    }

    // Main method to generate and save images
    public static void main(String[] args) {
        // Instantiate Generator
        Generator generator = new Generator();

        // Create a directory to save generated images
        String outputDir = "generated_images";
        new File(outputDir).mkdirs();

        // Generate and save images
        int numImagesToGenerate = 1;
        for (int i = 0; i < numImagesToGenerate; i++) {
            double[] noise = generator.generateNoise(NOISE_SIZE);
            double[] generatedImage = generator.generateImage(noise);
            generator.saveImage(generatedImage, outputDir, i);
        }
    }
}
