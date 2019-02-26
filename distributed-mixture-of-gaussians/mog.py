import numpy as np

class GaussianDistribution:
    def __init__(self, mean_vec, cov_mat):
        self.mean_vec = mean_vec
        self.dim, = self.mean_vec.shape
        self.cov_mat = cov_mat
        self.inverse_cov_mat = np.linalg.inv(self.cov_mat)
        # self.cov_mat_det = np.linalg.det(self.cov_mat)
        self.coef = (((2. * np.pi) ** self.dim) * np.linalg.det(self.cov_mat)) ** -.5

    def __call__(self, point):
        centered_point = point - self.mean_vec
        density = self.coef * np.exp(-.5 * centered_point.T @ self.inverse_cov_mat @ centered_point)

        return density

class MixtureOfGaussians:
    def __init__(self, gaussian_distributions, cluster_probs):
        self.gaussian_distributions = gaussian_distributions
        self.cluster_num = len(gaussian_distributions)
        self.cluster_probs = cluster_probs

    def calc_cond_probs(self, point):
        # [cluster_num]
        probs = np.array(
            [
                self.cluster_probs[z] * self.gaussian_distributions[z](point)
                for z in range(self.cluster_num)
            ]
        )
        probs /= probs.sum()

        return probs

