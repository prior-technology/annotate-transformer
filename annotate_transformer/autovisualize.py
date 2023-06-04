#Generated from codex - not useful but interesting
import torch
import matplotlib.pyplot as plt

def VisualizeEmbedding(tensor):
  """
  Reduce the dimensionality of the input vectors and plot the result
  """
  # Reduce the dimensionality of the input vectors
  pca = PCA(n_components=2)
  pca.fit(tensor)
  reduced_tensor = pca.transform(tensor)
  # Plot the result
  plt.scatter(reduced_tensor[:,0], reduced_tensor[:,1])
  plt.show()

class PCA:
    """Performs principal Component Analysis on a pytorch tensor"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, tensor):
        """
        Calculates the mean and eigenvalues and eigenvectors of the input tensor
        """
        # Calculate the mean of the tensor
        self.mean = torch.mean(tensor, dim=0)
        # Calculate the covariance matrix
        covariance_matrix = torch.zeros((tensor.shape[1], tensor.shape[1]))
        for i in range(tensor.shape[0]):
            covariance_matrix += (tensor[i] - self.mean).view(tensor.shape[1], 1) @ (tensor[i] - self.mean).view(1, tensor.shape[1])
        covariance_matrix = covariance_matrix / (tensor.shape[0] - 1)
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = torch.symeig(covariance_matrix, eigenvectors=True)