import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


class Model_view(nn.Module):
    def __init__(self, cluster_num, dim_in, dim_out):
        super(Model_view, self).__init__()
        self.layer_num = 4
        self.cluster = nn.Sequential(
            nn.Linear(dim_out, cluster_num),
            nn.Softmax(dim=1)
        )

        self.net = nn.Sequential(
            nn.Linear(dim_in, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, dim_out), nn.BatchNorm1d(dim_out), nn.ReLU(),
        )
        self.fc_1 = nn.Sequential(nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU())
        self.fc_2 = nn.Sequential(nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU())
        self.fc_3 = nn.Sequential(nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU())
        self.fc_4 = nn.Sequential(nn.Linear(dim_out, dim_out*2), nn.BatchNorm1d(dim_out*2), nn.ReLU())


    def getDistribution(self, index, h_feature):
        if index == 0:
            params, feature_dim = self.fc_1(h_feature), 1024
        elif index == 1:
            params, feature_dim = self.fc_2(h_feature), 1024
        elif index == 2:
            params, feature_dim = self.fc_3(h_feature), 1024
        elif index == 3:
            params = self.fc_4(h_feature)
            feature_dim = params.shape[1] // 2
        else:
            raise ValueError(f"Invalid index {index}")

        mu, sigma = params[:, :feature_dim], params[:, feature_dim:]

        sigma = softplus(sigma) + 1e-7
        h_distribution = Independent(Normal(loc=mu, scale=sigma), 1)
        return h_distribution.rsample(), h_distribution

    def forward(self, input):
        h_features, h_distributions = {}, {}
        x = input
        for index in range(self.layer_num):
            x = self.net[(index) * 3](x)
            x = self.net[(index) * 3 + 1](x)
            x = self.net[(index) * 3 + 2](x)
            h_features[index], h_distributions[index] = self.getDistribution(index, x)
        h_cluster = self.cluster(h_features[self.layer_num-1])
        return h_features, h_cluster, h_distributions


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, n_clusters, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size, self.temperature, self.device = batch_size, temperature, device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N)).fill_diagonal_(0)
        for i in range(batch_size): mask[i, batch_size + i] = mask[batch_size + i, i] = 0
        return mask.bool()
    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = F.normalize(torch.cat((z_i, z_j), dim=0), dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        positive_samples = torch.cat((torch.diag(sim, self.batch_size), torch.diag(sim, -self.batch_size)), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return self.criterion(logits, labels) - torch.log(torch.tensor(self.batch_size).float())

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num, self.temperature, self.device = class_num, temperature, device
        self.mask = self.mask_correlated_clusters()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)
    def mask_correlated_clusters(self):
        N = 2 * self.class_num
        mask = torch.ones((N, N), device=self.device).fill_diagonal_(0)
        for i in range(self.class_num): mask[i, self.class_num + i] = mask[self.class_num + i, i] = 0
        return mask.bool()
    def forward(self, c_i, c_j):
        N, c = 2 * self.class_num, F.normalize(torch.cat((c_i.t(), c_j.t()), dim=0), dim=1)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        positive_clusters = torch.cat((torch.diag(sim, self.class_num), torch.diag(sim, -self.class_num)), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        return self.criterion(logits, labels) - torch.log(torch.tensor(self.class_num).float())
