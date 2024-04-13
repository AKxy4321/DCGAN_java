import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class GeneratorModel {

    private static final int INPUT_SIZE = 100;
    private static final int HIDDEN_DIM = 7 * 7 * 256;
    private static final int FILTER1_SIZE = 5;
    //rivate static final int FILTER2_SIZE = 5;

    private final float[] weights1; // Dense layer weights (input -> hidden)
    private final float[] weights2; // Conv2DTranspose layer weights (hidden -> 7x7x128)
    private final float[] weights3; // Conv2DTranspose layer weights (128 -> 14x14x64)
    private final float[] weights4; // Conv2DTranspose layer weights (64 -> 28x28x1)

    public GeneratorModel(float[] weights1, float[] weights2, float[] weights3, float[] weights4) {
        this.weights1 = weights1;
        this.weights2 = weights2;
        this.weights3 = weights3;
        this.weights4 = weights4;
    }

    public float[] generate(float[] input) {
        float[] hidden = dense(input);
        float[] conv1 = conv2dTranspose(hidden, weights2, 7, 7, 128);
        float[] conv2 = conv2dTranspose(conv1, weights3, 14, 14, 64);
        return conv2dTranspose(conv2, weights4, 28, 28, 1);
    }

    private float[] dense(float[] input) {
        int numInputs = input.length;
        float[] hidden = new float[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float sum = 0.0f;
            for (int j = 0; j < numInputs; j++) {
                sum += input[j] * weights1[i * numInputs + j];
            }
            hidden[i] = leakyReLU(sum); // Apply LeakyReLU activation
        }
        return hidden;
    }

//    private float leakyReLU(float x) {
//        return Math.max(0.1f * x, x); // LeakyReLU with slope 0.1
//    }

    private float leakyReLU(float x) {
        // Removed Leaky ReLU for pure noise generation
        return x;
    }

    private float[] conv2dTranspose(float[] input, float[] weights, int outputWidth, int outputHeight, int numFilters) {
        int numInputs = input.length / (outputWidth * outputHeight);
        float[] output = new float[outputWidth * outputHeight * numFilters];
        for (int filter = 0; filter < numFilters; filter++) {
            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    float sum = 0.0f;
                    for (int inY = 0; inY < FILTER1_SIZE; inY++) {
                        for (int inX = 0; inX < FILTER1_SIZE; inX++) {
                            int inIndex = (y * outputWidth + x) * numInputs + inY * outputWidth + inX;
                            if (inY + y < outputHeight && inX + x < outputWidth) { // Handle padding (same)
                                sum += input[inIndex] * weights[filter * numInputs * FILTER1_SIZE * FILTER1_SIZE + inY * FILTER1_SIZE + inX];
                            }
                        }
                    }
                    output[(filter * outputHeight + y) * outputWidth + x] = leakyReLU(sum); // Apply LeakyReLU activation
                }
            }
        }
        return output;
    }

    public static void main(String[] args) {
        // Example weights (replace with actual weights from training)
        float[] weights1 = initializeWeights(INPUT_SIZE * HIDDEN_DIM);
        float[] weights2 = initializeWeights(HIDDEN_DIM * 7 * 7 * 128);
        float[] weights3 = initializeWeights(128 * 14 * 14 * 64);
        float[] weights4 = initializeWeights(64 * 28 * 28);

        GeneratorModel model = new GeneratorModel(weights1, weights2, weights3, weights4);

        // Generate random input noise
        float[] input = generateGaussianNoise();

        float[] output = model.generate(input);

        // Display some pixel values
        for (int i = 0; i < 10; i++) {
            System.out.println("Pixel value at index " + i + ": " + output[i]);
        }

        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int index = (y * 28) + x;
                float value = output[index]; // Assuming all channels have the same value
                int rgb = (int) ((value + 1) * 127.5f); // Scale to 0-255 range
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

    private static float[] initializeWeights(int size) {
        float[] weights = new float[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            weights[i] = (float) random.nextGaussian() * 0.15f; // Initialize with random values from normal distribution
        }
        return weights;
    }

    private static float[] generateGaussianNoise() {
        float[] noise = new float[GeneratorModel.INPUT_SIZE];
        Random random = new Random();
        for (int i = 0; i < GeneratorModel.INPUT_SIZE; i++) {
            noise[i] = (float) random.nextGaussian(); // Generate noise from normal distribution (between -1 and 1)
        }
        return noise;
    }
}