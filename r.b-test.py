import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_background(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to separate background and foreground
    # Adjust the threshold value as needed for your images
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Creating a mask for foreground
    foreground = cv2.bitwise_and(gray, gray, mask=thresholded)

    return foreground

def classify_brightness(foreground):
    # Calculate the histogram
    hist = cv2.calcHist([foreground], [0], None, [256], [0, 256])

    # Normalize the histogram
    hist = hist / hist.sum()

    # Calculate the average brightness
    avg_brightness = sum([i * hist[i] for i in range(256)])[0]
    print(f'The average brightness: {avg_brightness}')

    # Classify the brightness
    if avg_brightness < 60:
        brightness_class = "Low Brightness"
    elif avg_brightness > 170:
        brightness_class = "High Brightness"
    else:
        brightness_class = "Normal Brightness"

    return brightness_class, hist

# Path to your image
image_path = 'Low-164.jpg'
# image_path = 'Normal-164.jpg'

# Remove background and get foreground
foreground = remove_background(image_path)

# Classify brightness of the foreground
brightness_class, hist = classify_brightness(foreground)
print(f"The image is classified as: {brightness_class}")

# Plotting the histogram
plt.plot(hist)
plt.title('Histogram of Foreground Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
