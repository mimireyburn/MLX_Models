# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os
from PIL import Image



# %%

# Define the path to your custom dataset folder
#/root/autoimages/image.jpg
custom_data_path = "../autoimages"

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load your custom dataset using ImageFolder
custom_dataset = ImageFolder(root=custom_data_path, transform=transform)

# Split the custom dataset into train and test sets (you can adjust the split ratios as needed)
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

# Create DataLoader instances for both train and test sets
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Update the input channels to 3
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),  # Update the output channels to 8
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),  # Update the output channels to 8
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Update the output channels to 3
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Step 4: Define the loss function and optimizer
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# %%

# Step 5: Train the autoencoder
num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        inputs, _ = data
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed!")

# %%
# Step 6: Visualize the results
# Compare the original images with their reconstructed counterparts
def imshow(img, title):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Get a batch of test data
images, _ = next(iter(test_loader))

# Reconstruct images using the trained autoencoder
reconstructed = autoencoder(images)

# Visualize the original and reconstructed images
imshow(torchvision.utils.make_grid(images[0]), 'Original Images')
imshow(torchvision.utils.make_grid(reconstructed.detach()[0]), 'Reconstructed Images')

# %%
import random

embeddings = []

data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

custom_dataset = ImageFolder(root=custom_data_path, transform=transform)

autoencoder.encoder.eval()

with torch.no_grad():
    for images, _ in data_loader:
        embeddings_batch = autoencoder.encoder(images)
        embeddings.append(embeddings_batch.view(embeddings_batch.size(0), -1))

# Concatenate the embeddings and convert them to a numpy array
raw_embeddings = embeddings
embeddings = torch.cat(embeddings, dim=0)
embeddings_np = embeddings.numpy()

# Sample a subset of the embeddings to reduce the number of data points
subset_size = 10000  # Set the number of data points you want to plot
if embeddings_np.shape[0] > subset_size:
    random_indices = random.sample(range(embeddings_np.shape[0]), subset_size)
    embeddings_np_subset = embeddings_np[random_indices]
else:
    embeddings_np_subset = embeddings_np

# Apply PCA on the subset of embeddings
n_components = 3  # Set the number of components you want to keep (change to 3)
pca = PCA(n_components=n_components)
# pca_result = pca.fit_transform(embeddings_np_subset)
pca_result = pca.fit_transform(embeddings_np)


# %%
# Create a 2D scatter plot of Principal Component 1 and Principal Component 2
plt.figure(figsize=(10, 8))

# Scatter plot using the first two principal components
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization with Two Components')
plt.show()


# %%
# print(pca_result.shape)
# print(embeddings_np.shape)

# Given conditions
target_pca1 = -2.5
target_pca2 = 1.0
target_pca3 = 0.0
tolerance = 0.25  # Adjust the tolerance value based on your desired closeness to the targets

# Find indices where PCA1, PCA2, and PCA3 are close to the target values
indices = np.where(
    (np.abs(pca_result[:, 0] - target_pca1) <= tolerance) &
    (np.abs(pca_result[:, 1] - target_pca2) <= tolerance) &
    (np.abs(pca_result[:, 2] - target_pca3) <= tolerance)
)[0]

# Randomly choose one index from the matching indices
if len(indices) > 0:
    print(len(indices))
    selected_indices = []
    for x in range(5):
        selected_indices.append(np.random.choice(indices))
    print(f"The random datapoint is at index: {selected_indices}")
    # Access the corresponding data in pca_result
    random_datapoint = pca_result[selected_index]
    print(f"PCA1: {random_datapoint[0]}, PCA2: {random_datapoint[1]}, PCA3: {random_datapoint[2]}")
else:
    print("No datapoint found that meets the criteria.")

# Step 7: Display the original and encoded images
# selected_image, _ = custom_dataset[selected_index]  # Access the original image
# selected_image = selected_image.unsqueeze(0)  # Add batch dimension for processing

# Pass the original image through the encoder to get the encoded representation
# with torch.no_grad():
    # reconstructed = autoencoder(selected_image)

# # Decode the encoded representation to get the reconstructed image
# with torch.no_grad():
#     reconstructed_image = autoencoder.decoder(encoded_representation)

for selected_image in selected_indices: 

    selected_image, _  = custom_dataset[selected_image] 
    selected_image = selected_image.unsqueeze(0)

    # Unnormalize the images
    selected_image = selected_image / 2 + 0.5


    # Plot the images side by side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(selected_image.squeeze().numpy(), (1, 2, 0)))
    plt.title('Original Image')
    plt.axis('off')

    plt.show()



# %%
# print(pca_result.shape)
# print(embeddings_np.shape)

# Given conditions
target_pca1 = 12.0
target_pca2 = 2.0
# target_pca3 = 0.0
tolerance = 0.25  # Adjust the tolerance value based on your desired closeness to the targets

# Find indices where PCA1, PCA2, and PCA3 are close to the target values
indices = np.where(
    (np.abs(pca_result[:, 0] - target_pca1) <= tolerance) &
    (np.abs(pca_result[:, 1] - target_pca2) <= tolerance) 
)[0]

# Randomly choose one index from the matching indices
if len(indices) > 0:
    print(len(indices))
    selected_indices = []
    for x in range(5):
        selected_indices.append(np.random.choice(indices))
    print(f"The random datapoint is at index: {selected_indices}")
    # Access the corresponding data in pca_result
    random_datapoint = pca_result[selected_index]
    print(f"PCA1: {random_datapoint[0]}, PCA2: {random_datapoint[1]}, PCA3: {random_datapoint[2]}")
else:
    print("No datapoint found that meets the criteria.")

# Step 7: Display the original and encoded images
# selected_image, _ = custom_dataset[selected_index]  # Access the original image
# selected_image = selected_image.unsqueeze(0)  # Add batch dimension for processing

# Pass the original image through the encoder to get the encoded representation
# with torch.no_grad():
    # reconstructed = autoencoder(selected_image)

# # Decode the encoded representation to get the reconstructed image
# with torch.no_grad():
#     reconstructed_image = autoencoder.decoder(encoded_representation)

for selected_image in selected_indices: 

    selected_image, _  = custom_dataset[selected_image] 
    selected_image = selected_image.unsqueeze(0)

    # Unnormalize the images
    selected_image = selected_image / 2 + 0.5


    # Plot the images side by side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(selected_image.squeeze().numpy(), (1, 2, 0)))
    plt.title('Original Image')
    plt.axis('off')

    plt.show()


# %%
import plotly.graph_objects as go

# Create an interactive 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    z=pca_result[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        opacity=0.7,
        colorscale='Viridis',  # Choose a colorscale of your preference
    )
)])

fig.update_layout(
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3',
    ),
    title='Interactive PCA Visualization with Three Components',
    width=1200,  # Set the width of the plot
    height=800,  # Set the height of the plot
)

# Show the interactive plot in the notebook or save it as an HTML file
fig.show()

# %%



