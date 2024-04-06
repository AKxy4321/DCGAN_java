// Partial Differentiation in Java

import java.lang.Math;

public class PartialDifferentiation {

    // Define the function f(x, y) = x^2 + y^3
    public static double function(double x, double y) {
        return Math.pow(x, 2) + Math.pow(y, 3);
    }

    // Compute the partial derivative of f(x, y) with respect to x using central difference
    public static double partialDerivativeX(double x, double y, double h) {
        double xPlusH = x + h;
        double xMinusH = x - h;
        return (function(xPlusH, y) - function(xMinusH, y)) / (2 * h);
    }

    // Compute the partial derivative of f(x, y) with respect to y using central difference
    public static double partialDerivativeY(double x, double y, double h) {
        double yPlusH = y + h;
        double yMinusH = y - h;
        return (function(x, yPlusH) - function(x, yMinusH)) / (2 * h);
    }

    public static void main(String[] args) {
        double x = 2.0;
        double y = 3.0;
        double h = 0.0001; // Step size for finite differences

        // Compute partial derivatives
        double partialX = partialDerivativeX(x, y, h);
        double partialY = partialDerivativeY(x, y, h);

        // Print results
        System.out.println("Partial derivative with respect to x: " + partialX);
        System.out.println("Partial derivative with respect to y: " + partialY);
    }
}
