import java.util.Random;

public class Discriminator {
    // Discriminator network parameters
    private double[][] discWeights1;
    private double[][] discWeights2;
    private double[] discBias1;
    private double[] discBias2;

    private double LEARNING_RATE;
    private Random random;

    public Discriminator(int imageSize, double learningRate) {
        LEARNING_RATE = learningRate;
        random = new Random();

        // Initialize discriminator network parameters randomly
        int inputSize = imageSize * imageSize;
        int hiddenSize = 128;
        discWeights1 = randomWeights(inputSize, hiddenSize);
        discBias1 = randomBiases(hiddenSize);
        discWeights2 = randomWeights(hiddenSize, 1);
        discBias2 = randomBiases(1);
    }

    // Discriminator network forward pass
//    public double predictReal(double[] image) {
//        double[] hiddenLayer = MatrixUtils.sigmoid(MatrixUtils.add(MatrixUtils.dot(image, MatrixUtils.transpose(discWeights1)), discBias1));
//        return MatrixUtils.sigmoid(MatrixUtils.add(MatrixUtils.dot(hiddenLayer, MatrixUtils.transpose(discWeights2)), discBias2))[0];
//    }

    // Update discriminator network parameters
    public void updateParameters(double[] image, double[] hiddenDelta, double[] outputDelta) {
        // Update discWeights2 and discBias2
        for (int i = 0; i < discWeights2.length; i++) {
            for (int j = 0; j < discWeights2[0].length; j++) {
                discWeights2[i][j] -= LEARNING_RATE * hiddenDelta[j] * outputDelta[0];
            }
        }
        for (int i = 0; i < discBias2.length; i++) {
            discBias2[i] -= LEARNING_RATE * outputDelta[0];
        }

        // Update discWeights1 and discBias1
        for (int i = 0; i < discWeights1.length; i++) {
            for (int j = 0; j < discWeights1[0].length; j++) {
                discWeights1[i][j] -= LEARNING_RATE * image[i] * hiddenDelta[j];
            }
        }
        for (int i = 0; i < discBias1.length; i++) {
            discBias1[i] -= LEARNING_RATE * hiddenDelta[i];
        }
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
}