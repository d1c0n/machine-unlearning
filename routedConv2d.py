import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RoutedConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_filters, stride=1, padding=0
    ):
        super(OptimizedRoutedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding

        # Create weight and bias for convolution
        self.weights = [
            nn.Parameter(
                torch.Tensor(1, in_channels, kernel_size, kernel_size)
                for _ in range(out_channels)
            )
        ]
        self.biases = nn.Parameter(torch.Tensor(n_filters))

        # Initialize weights and biases
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)

        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

        # Create the router (a linear layer)
        self.router = nn.Linear(in_channels, out_channels)

        # Initialize filter usage tracker
        self.register_buffer("filter_usage", torch.zeros(out_channels))

    def forward(self, x, labels):
        batch_size, _, height, width = x.shape

        # Use the router to decide which filters to use
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        router_output = self.router(pooled)

        aux_loss = load_balancing_loss_func(
            router_output, self.out_channels, self.n_filters
        )

        # Get the indices of the top n_filters
        _, top_indices = torch.topk(router_output, self.n_filters, dim=1)

        # Create a list to store the convolution results
        conv_weights = torch.stack(self.weights, dim=0)
        conv_biases = self.biases[top_indices]

        x = F.conv2d(
            x,
            conv_weights,
            conv_biases,
            stride=self.stride,
            padding=self.padding,
        )

        loss = F.cross_entropy(x, labels)

        return x


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class OptimizedRoutedConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_filters, stride=1, padding=0
    ):
        super(OptimizedRoutedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding

        # Create weight and bias for convolution
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # Create the router (a linear layer)
        self.router = nn.Linear(in_channels, out_channels)

        # Initialize filter usage tracker
        self.register_buffer("filter_usage", torch.zeros(out_channels))

    def forward(self, x):
        batch_size = x.shape[0]

        # Use the router to decide which filters to use
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        router_output = self.router(pooled)

        # Get the indices of the top n_filters
        _, top_indices = torch.topk(router_output, self.n_filters, dim=1)

        # Create a mask for the selected filters
        mask = torch.zeros(batch_size, self.out_channels, 1, 1, device=x.device)
        mask.scatter_(1, top_indices.unsqueeze(-1).unsqueeze(-1), 1)

        # Update filter usage
        self.filter_usage += mask.sum(dim=(0, 2, 3))

        # Perform convolution for all filters
        conv_output = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        # Apply the mask to keep only the selected filters' output
        return conv_output * mask

    def get_load_balancing_loss(self, batch_size):
        # Compute the ideal uniform usage
        ideal_usage = batch_size * self.n_filters / self.out_channels

        # Compute the deviation from ideal usage
        usage_deviation = self.filter_usage - ideal_usage

        # Compute the load balancing loss (e.g., using mean squared error)
        load_balancing_loss = torch.mean(usage_deviation**2)

        return load_balancing_loss

    def reset_filter_usage(self):
        self.filter_usage.zero_()


class RoutedConv2dWithUnlearning(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_filters, stride=1, padding=0
    ):
        super(RoutedConv2dWithUnlearning, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.router = nn.Linear(in_channels, out_channels)

        self.register_buffer("filter_usage", torch.zeros(out_channels))
        self.register_buffer(
            "class_filter_usage", torch.zeros(10, out_channels)
        )  # Assuming CIFAR10 (10 classes)

    def forward(self, x, target=None):
        batch_size = x.shape[0]

        pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        router_output = self.router(pooled)

        _, top_indices = torch.topk(router_output, self.n_filters, dim=1)

        mask = torch.zeros(batch_size, self.out_channels, 1, 1, device=x.device)
        mask.scatter_(1, top_indices.unsqueeze(-1).unsqueeze(-1), 1)

        self.filter_usage += mask.sum(dim=(0, 2, 3))

        if target is not None:
            for i in range(batch_size):
                self.class_filter_usage[target[i]] += mask[i, :, 0, 0]

        conv_output = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return conv_output * mask

    def get_load_balancing_loss(self, batch_size):
        ideal_usage = batch_size * self.n_filters / self.out_channels
        usage_deviation = self.filter_usage - ideal_usage
        return torch.mean(usage_deviation**2)

    def reset_filter_usage(self):
        self.filter_usage.zero_()
        self.class_filter_usage.zero_()

    def get_class_filters(self, class_idx, threshold=0.8):
        class_usage = self.class_filter_usage[class_idx]
        total_usage = class_usage.sum()
        sorted_indices = torch.argsort(class_usage, descending=True)
        cumulative_usage = torch.cumsum(
            class_usage[sorted_indices] / total_usage, dim=0
        )
        return sorted_indices[cumulative_usage <= threshold]


class UnlearningOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, filters_to_unlearn=None):
        defaults = dict(lr=lr, filters_to_unlearn=filters_to_unlearn)
        super(UnlearningOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if group["filters_to_unlearn"] is not None and hasattr(
                    p, "filter_index"
                ):
                    mask = torch.ones_like(d_p)
                    mask[group["filters_to_unlearn"]] = 0
                    d_p = d_p * mask

                p.data.add_(d_p, alpha=-group["lr"])

        return loss


def unlearning_procedure(
    model, unlearn_class, unlearn_dataloader, num_epochs=5, lr=0.001
):
    filters_to_unlearn = []
    for module in model.modules():
        if isinstance(module, RoutedConv2dWithUnlearning):
            filters = module.get_class_filters(unlearn_class)
            filters_to_unlearn.append(filters)

    optimizer = UnlearningOptimizer(
        model.parameters(), lr=lr, filters_to_unlearn=filters_to_unlearn
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in unlearn_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Reset filter usage statistics after unlearning
    for module in model.modules():
        if isinstance(module, RoutedConv2dWithUnlearning):
            module.reset_filter_usage()


# Usage example
# model = YourModel()  # This should include RoutedConv2dWithUnlearning layers
# unlearning_procedure(model, unlearn_class=1, unlearn_dataloader=class_1_dataloader)
