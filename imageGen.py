

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import json

# Specify the path to your image dataset folder
dataset_path = "./JPEG"

# Load images from the dataset folder
image_paths = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(('.jpg', '.png'))]
image_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]  # Extract image names without extensions

# Read images and convert them to grayscale (adjust as needed)
images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_paths]

# Flatten the images into vectors
n_samples, n_features = len(images), images[0].size
data = np.array(images).reshape((n_samples, n_features))

# Scale the data
data = StandardScaler().fit_transform(data)

# Reduce dimensionality with PCA (optional)
n_components = 2
pca = PCA(n_components=n_components)
data = pca.fit_transform(data)

# Perform k-means clustering with 12 clusters
n_clusters = 12  # You can change this to the desired number of clusters
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)

# Assign cluster labels to the original data
labels = kmeans.labels_

# Plot the results
fig, ax = plt.subplots(figsize=(6, 6))

scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')

# Create a list of image paths corresponding to each data point
image_paths_per_point = [image_paths[i] for i in range(n_samples)]
image_names_per_point = [image_names[i] for i in range(n_samples)]

mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(f"Cluster: {labels[sel.target.index]}, Image Name: {image_names_per_point[sel.target.index]}"))

# Initialize a dictionary to store image information
image_data = {'images': []}

def on_hover(sel):
    index = sel.target.index
    
    # Compute distances to find the four closest data points
    distances = np.linalg.norm(data - data[index], axis=1)
    closest_indices = np.argsort(distances)[1:5]  # Exclude the selected data point itself
    
    # Overlay the images on top of each other with varying opacities
    composite_image = np.zeros_like(images[0], dtype=np.float32)
    
    for i, closest_index in enumerate(closest_indices):
        opacity = 1 / (i + 2)  # Opacity decreases for subsequent images
        
        
        enhanced_image = np.clip((images[closest_index].astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
        
        # Overlay the images with varying opacities
        composite_image = cv2.addWeighted(composite_image, 1, enhanced_image.astype(np.float32), opacity, 0)
    
    # Boost contrast of the final composite image if needed
    #final_composite_image = np.clip(contrast_factor * (composite_image - 128) + 128, 0, 255).astype(np.uint8)
    # Adjust the contrast factor to control the contrast
    contrast_factor = .5  # You can experiment with different values

    # Boost the contrast
    final_composite_image = np.clip(contrast_factor * (composite_image - 128) + 128, 0, 255).astype(np.uint8)
    # Display the overlaid images directly in the main plot
    ax.plot(1080,1080)
    ax.imshow(final_composite_image, cmap='gray', alpha=1, extent=(data[:, 0].min(), data[:, 0].max(), data[:, 0].min(), data[:, 0].max()))    # Save image information to the dictionary
    image_info = {
        'name': image_names_per_point[index],
        'x_pos': data[index, 0],
        'y_pos': data[index, 1],
        'adj_images': [image_names_per_point[i] for i in closest_indices]
    }
    image_data['images'].append(image_info)

# Connect the on_hover function to the main plot
mplcursors.cursor(hover=True).connect("add", on_hover)

plt.title('K-Means Clustering of Images')
plt.xlabel(f'Principal Component 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}% variance explained)')
plt.ylabel(f'Principal Component 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}% variance explained)')

# Set the limits to display the entire plot
ax.set_xlim(data[:, 0].min(), data[:, 0].max())
ax.set_ylim(data[:, 0].min(), data[:, 0].max())
ax.set_facecolor((1, 1, 1, 0))  # (R, G, B, alpha)

# Save image information to the dictionary for all data points
for index in range(n_samples):
    # Compute distances to find the four closest data points
    distances = np.linalg.norm(data - data[index], axis=1)
    closest_indices = np.argsort(distances)[1:5]  # Exclude the selected data point itself
    
    image_info = {
        'name': image_names_per_point[index],
        'x_pos': data[index, 0],
        'y_pos': data[index, 1],
        'adj_images': [image_names_per_point[i] for i in closest_indices],
        'opacity':0
    }
    image_data['images'].append(image_info)

# Save image data to a JSON file
with open('image_data.json', 'w') as json_file:
    json.dump(image_data, json_file, indent=4)


plt.show()
plt.set_facecolor((1, 1, 1, 0))  # (R, G, B, alpha)