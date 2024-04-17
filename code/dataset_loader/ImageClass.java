package dataset_loader;

import java.awt.image.BufferedImage;

public class ImageClass {
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