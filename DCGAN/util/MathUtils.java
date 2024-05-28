package DCGAN.util;

public class MathUtils {

    public static double l2Norm(double[][][][] array) {
        double l2_norm = 0;
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        l2_norm += array[i][j][k][l] * array[i][j][k][l];

        l2_norm = Math.sqrt(l2_norm);
        return l2_norm;
    }

    public static double l2Norm(double[][][] array) {
        double l2_norm = 0;
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    l2_norm += array[i][j][k] * array[i][j][k];

        l2_norm = Math.sqrt(l2_norm);
        return l2_norm;
    }

    public static double l2Norm(double[] array) {
        double l2_norm = 0;

        for (int i = 0; i < array.length; i++)
            l2_norm += array[i] * array[i];

        l2_norm = Math.sqrt(l2_norm);
        return l2_norm;
    }


    public static double mean(double[] array) {
        double sum = 0;
        for (double genLoss : array) {
            sum += genLoss;
        }
        return sum / array.length;
    }
    public static double[][] transpose(double[][] array) {
        double[][] transposed = new double[array[0].length][array.length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                transposed[j][i] = array[i][j];
        return transposed;
    }

    public static double[][][] transpose(double[][][] array) {
        double[][][] transposed = new double[array.length][array[0][0].length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    transposed[i][k][j] = array[i][j][k];
        return transposed;
    }

    public static double dotProduct(double[] array1, double[] array2) {
        double dotProduct = 0;
        for (int i = 0; i < array1.length; i++)
            dotProduct += array1[i] * array2[i];
        return dotProduct;
    }

    public static double dotProduct(double[][][] array1, double[][][] array2) {
        double dotProduct = 0;
        for (int i = 0; i < array1.length; i++)
            for (int j = 0; j < array1[0].length; j++)
                for (int k = 0; k < array1[0][0].length; k++)
                    dotProduct += array1[i][j][k] * array2[i][j][k];
        return dotProduct;
    }
}
