import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # Mean of latent distribution
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # Log-variance of latent distribution
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)    # Standard deviation
        eps = torch.randn_like(std)      # Epsilon ~ N(0,1)
        return mu + eps * std            # Sampled latent vector
    
    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))   # Flatten the input
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL Divergence between the learned latent distribution and standard normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# Training function
def train_vae(model, dataloader, optimizer, device, epochs=10):
    model.train()
    train_losses = []
    
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item() / len(data):.4f}')
        
        avg_loss = train_loss / len(dataloader.dataset)
        train_losses.append(avg_loss)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    
    return train_losses

# Visualization functions
def visualize_reconstructions(model, dataloader, device, num_images=8):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(dataloader))
        data = data.to(device)
        recon, _, _ = model(data)
        
        # Select the first num_images
        data = data[:num_images]
        recon = recon.view(-1, 1, 28, 28)[:num_images]
        
        # Move to CPU and convert to numpy
        data = data.cpu().numpy()
        recon = recon.cpu().numpy()
        
        # Plot
        plt.figure(figsize=(num_images * 2, 4))
        for i in range(num_images):
            # Original images
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(data[i].reshape(28, 28), cmap='gray')
            plt.title("Original")
            plt.axis('off')
            
            # Reconstructed images
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(recon[i].reshape(28, 28), cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')
        plt.show()

def visualize_latent_space(model, dataloader, device, num_batches=1000):
    model.eval()
    mu_list = []
    labels_list = []
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            mu_list.append(mu.cpu())
            labels_list.append(labels)
            if i >= num_batches:
                break
    
    mu = torch.cat(mu_list)
    labels = torch.cat(labels_list)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.show()

# Main function to run the training and visualization
def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    epochs = 10
    hidden_dim = 400
    latent_dim = 20
    
    # Data loading
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the VAE
    model = VAE(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the VAE
    train_losses = train_vae(model, train_loader, optimizer, device, epochs=epochs)
    
    # Plot training loss
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs + 1), train_losses, marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()
    
    # Visualize reconstructions
    visualize_reconstructions(model, test_loader, device, num_images=8)
    
    # Visualize latent space (only first two dimensions for visualization)
    if latent_dim >= 2:
        visualize_latent_space(model, test_loader, device, num_batches=1000)
    else:
        print("Latent dimension less than 2, skipping latent space visualization.")

if __name__ == "__main__":
    main()
