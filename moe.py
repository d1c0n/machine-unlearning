import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SparseMoEConvBlockWeighted(nn.Module):
    def __init__(
        self,
        router_dim,
        in_channels,
        out_channels,
        num_experts=8,
        top_k=2,
        kernel_size=3,
        stride=1,
        padding=1,
        device="cuda",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.router_dim = router_dim
        self.top_k = top_k
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.router = nn.Linear(router_dim, num_experts, bias=False)
        nn.init.xavier_uniform_(self.router.weight)
        self.experts = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape
        router_logits = self.router(x.view(batch_size, -1))
        router_output = nn.functional.softmax(router_logits, dim=1)

        # Get the average router choice
        router_avg = torch.mean(router_output, dim=0)

        # Create the target uniform distribution for the router
        target = torch.tensor([1 / self.num_experts] * self.num_experts).to(self.device)
        criterion = nn.MSELoss()
        router_loss = criterion(router_avg, target)

        routing_weights, selected_experts = torch.topk(router_output, self.top_k)

        expert_outputs = torch.zeros(batch_size, self.out_channels, height, width).to(
            x.device
        )

        for i, expert_idx in enumerate(selected_experts[0]):
            expert_layer = self.experts[expert_idx]
            # Select the corresponding routing weights for the expert
            routing_weights_expert = routing_weights[0, i]

            out = expert_layer(x)

            # Multiply the output by the routing weights (how strong the choice was)
            out = out * routing_weights_expert.unsqueeze(-1).unsqueeze(-1)

            expert_outputs += out

        return expert_outputs, router_loss


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 12, 3, 1, 1)
        self.fc1 = nn.Linear(12 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 12 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RoutedCNN(nn.Module):
    def __init__(self, device="cuda"):
        super(RoutedCNN, self).__init__()
        self.conv1 = SparseMoEConvBlockWeighted(
            router_dim=3 * 32 * 32,
            in_channels=3,
            out_channels=24,
            num_experts=10,
            top_k=1,
        )
        self.fc1 = nn.Linear(24 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)
        self.device = device

    def forward(self, x, return_router_loss=False):
        x, router_loss_1 = self.conv1(x)

        x = x.view(-1, 24 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if return_router_loss:
            return x, router_loss_1
        else:
            return x
