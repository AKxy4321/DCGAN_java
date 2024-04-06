package Ziroh;

import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ImageDatasetLoader {

    private static final int IMAGE_WIDTH = 224;
    private static final int IMAGE_HEIGHT = 224;
    private static final int NUM_CHANNELS = 3; // Assuming RGB images

    private static final Logger LOGGER = Logger.getLogger(ImageDatasetLoader.class.getName());

    public static List<ImageClass> loadDataset(String datasetPath) {
        List<ImageClass> dataset = new ArrayList<>();

        File datasetDir = new File(datasetPath);
        if (!datasetDir.isDirectory()) {
            LOGGER.log(Level.SEVERE, "Dataset path must be a directory: {0}", datasetPath);
            return dataset;
        }

        File[] classDirs = datasetDir.listFiles(File::isDirectory);
        if (classDirs == null) {
            LOGGER.log(Level.SEVERE, "Failed to list class directories in dataset: {0}", datasetPath);
            return dataset;
        }

        for (File classDir : classDirs) {
            String className = classDir.getName();
            File[] imageFiles = classDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".jpeg") || name.toLowerCase().endsWith(".png"));
            if (imageFiles == null) {
                LOGGER.log(Level.SEVERE, "Failed to list image files in class directory: {0}", classDir.getAbsolutePath());
                continue;
            }
            for (File imageFile : imageFiles) {
                try {
                    BufferedImage image = ImageIO.read(imageFile);
                    if (image == null) {
                        LOGGER.log(Level.WARNING, "Failed to read image file: {0}", imageFile.getAbsolutePath());
                        continue;
                    }
                    if (image.getWidth() != IMAGE_WIDTH || image.getHeight() != IMAGE_HEIGHT) {
                        LOGGER.log(Level.WARNING, "Image dimensions or type not as expected: {0}", imageFile.getAbsolutePath());
                        continue;
                    }
                    dataset.add(new ImageClass(className, image));
                } catch (IOException e) {
                    LOGGER.log(Level.SEVERE, "Error reading image file: {0}", imageFile.getAbsolutePath());
                }
            }
        }

        return dataset;
    }

    public static void main(String[] args) {
        try {
            String datasetPath = "Ziroh/dataset/test"; // Specify the path to your dataset folder
            List<ImageClass> dataset = loadDataset(datasetPath);
            LOGGER.log(Level.INFO, "Loaded {0} images from dataset", dataset.size());
            // Now you have a list of ImageClass objects containing images and their respective class names
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error loading dataset: {0}", e.getMessage());
        }
    }
}

class ImageClass {
    private String className;
    private BufferedImage image;

    public ImageClass(String className, BufferedImage image) {
        this.className = className;
        this.image = image;
    }

    public String getClassName() {
        return className;
    }

    public BufferedImage getImage() {
        return image;
    }
}
