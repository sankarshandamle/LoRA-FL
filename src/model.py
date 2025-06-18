import torch
import torch.nn as nn


import sys


### Standard MLP

class Net(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64, bias=False)  # First hidden layer without bias
        self.bn1 = nn.BatchNorm1d(64)                      # Batch normalization
        self.fc2 = nn.Linear(64, 32, bias=False)           # Second hidden layer without bias
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1, bias=False)            # Output layer without bias
        self.dropout = nn.Dropout(0.5)                     # Dropout for regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))  # Apply batch normalization after each linear layer
        x = self.dropout(x)                    # Apply dropout after activation
        x = torch.relu(self.bn2(self.fc2(x)))  # Apply batch normalization after each linear layer
        x = self.dropout(x)                    # Apply dropout after activation
        x = self.fc3(x)                        # Output layer
        return x

### LoRA Adapter

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super(LoRALinear, self).__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)  # Main weight layer

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False  # Ensure base weights are not trainable

        # LoRA-specific low-rank matrices
        self.A = nn.Parameter(torch.randn(in_features, rank) * 1e-8)  # Small random initialization
        self.B = nn.Parameter(torch.randn(rank, out_features) * 1e-8)

        self.alpha = alpha  # Scaling factor

    def forward(self, x):
        return self.base_layer(x) + self.alpha * (x @ self.A @ self.B)

class LoRANet(nn.Module):
    def __init__(self, input_size: int, rank: int) -> None:
        super(LoRANet, self).__init__()
        self.fc1 = LoRALinear(input_size, 64, rank)  # First LoRA-enhanced hidden layer
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = LoRALinear(64, 32, rank)          # Second LoRA-enhanced hidden layer
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = LoRALinear(32, 1, rank)           # Output layer
        self.dropout = nn.Dropout(0.5)

    def normalize_lora_weights(self):
        """Normalizes LoRA weight matrices A and B to prevent numerical instability."""
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                if module.A.norm().item() > 0:  # Avoid division by zero
                    module.A.data /= module.A.norm()
                if module.B.norm().item() > 0:
                    module.B.data /= module.B.norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.normalize_lora_weights()  # Ensure stability
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

class DisparateImpactMaximizerLoss(nn.Module):
    def __init__(self, weight=0.5, epsilon=1e-5, measure = 'eo'):
        super(DisparateImpactMaximizerLoss, self).__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.measure = measure

    def forward(self, logits, targets, sensitive_feature):
        logits = logits.squeeze()

        # Apply sigmoid and clamp to avoid log(0) or log(1)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.epsilon, max=1 - self.epsilon)

        # Convert sensitive_feature ('Male', 'Female') to 0 and 1
        sensitive_feature_tensor = torch.tensor([1 if x == 1 else 0 for x in sensitive_feature], dtype=torch.float, device="cuda")

        # Create masks for sensitive feature (Male = 1, Female = 0)
        male_mask = sensitive_feature_tensor  # 1 for Male, 0 for Female
        female_mask = 1 - male_mask  # 1 for Female, 0 for Male

        # Calculate log loss
        log_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        # Calculate loss for each group (and prevent NaN issues from empty groups)
        male_mask_sum = male_mask.sum() + self.epsilon
        female_mask_sum = female_mask.sum() + self.epsilon



        #### DemP ####
        male_p  = (male_mask * probs).sum() / male_mask_sum
        female_p = (female_mask * probs).sum() / female_mask_sum

        demp = torch.abs(male_p - female_p)

        #### EO #####
        male_positive_sum = (male_mask * targets).sum()
        female_positive_sum = (female_mask * targets).sum()
        male_fnr = (male_mask * (1 - probs) * targets).sum() / male_positive_sum
        female_fnr = (female_mask * (1 - probs) * targets).sum() / female_positive_sum

        male_negative_sum = (male_mask * (1 - targets)).sum()
        female_negative_sum = (female_mask * (1 - targets)).sum()
        male_fpr = (male_mask * probs * (1 - targets)).sum() / male_negative_sum
        female_fpr = (female_mask * probs * (1 - targets)).sum() / female_negative_sum
        eo = torch.max(torch.abs(male_fnr - female_fnr), torch.abs(male_fpr - female_fpr))


        male_loss = (male_mask * log_loss).sum() / male_mask_sum
        female_loss = (female_mask * log_loss).sum() / female_mask_sum


        male_loss = (male_mask * log_loss).sum() / male_mask_sum
        female_loss = (female_mask * log_loss).sum() / female_mask_sum

        # Calculate absolute difference and normalize between 0 and 1
        abs_diff = torch.abs(male_loss - female_loss)

        # Normalize the difference to range [0, 1] using min_max_range
        min_range, max_range = torch.min(male_loss, female_loss), torch.max(male_loss, female_loss)
        normalized_diff = (abs_diff - min_range) / (max_range - min_range)

        # Clamp the normalized difference to ensure it stays between 0 and 1
        normalized_diff = torch.clamp(normalized_diff, min=0.0, max=1.0)

        # Inverse the normalized difference and calculate the final loss
        # loss = 1 / (normalized_diff + self.epsilon)

        if self.measure == 'eo':
            loss = -torch.abs(eo)
        elif self.measure == 'dp':
            loss = -torch.abs(demp)
        else:
            loss = -torch.abs(eo)



        # Debugging: Check for NaN or Inf values
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("⚠️ NaN detected in loss!")
            print(f"Loss: {loss}")
            print(f"Probs: {probs}")
            print(f"Male Mask Sum: {male_mask_sum}")
            print(f"Female Mask Sum: {female_mask_sum}")
            print(f"male_loss: {male_loss}")
            print(f"female_loss: {female_loss}")
            print(f"log_loss: {log_loss}")
            sys.exit(0)

        return loss
    

class CustomTwoPhaseLowRankLoss(nn.Module):
    def __init__(self):
        """
        Initializes the loss function with a given low-rank factorization rank.
        """
        super(CustomTwoPhaseLowRankLoss, self).__init__()
        self.lam = 0.5
        self.epsilon = 1e-8


    def Phase1Regularizer(self, theta_g, theta_i, theta_lora, logits, targets, sensitive_labels):
        """
        Computes the loss for low-rank adaptation.

        Args:
            theta_g: Global model parameters (k-1)th.
            theta_i: Benign Local model parameters (k)th.
            theta_g - theta_i : \Delta \theta for the kth round
            theta_lora: Adversarial LoRA-enhanced model parameters.
            logits: Tensor of shape (N, C), raw model outputs before softmax.
            targets: Tensor of shape (N,), correct class indices.
            sensitive_labels: List of sensitive labels for each data point.

        Returns:
            A scalar loss value combining cross-entropy and low-rank regularization.
        """

        """
        Loss1: AB^T - theta_i - theta_g
        """

        """
        \delta theta = theta_i - theta_g
        \delta theta = AB^T
        output: theta_g + AB^T
        """

        ## first compute theta_i - theta_g
        added_params = {
            name.split('.')[0]: param1 - param2
            for ((name, param1), (_, param2)) in zip(theta_g.named_parameters(), theta_i.named_parameters())
            if 'fc' in name  # Check for 'fc' to include fc1, fc2, and fc3 layers
        }

        ## Do AB^T - added_params
        # Compute AB^T for each LoRALinear layer in l1
        lora_params = {}
        for name, module in theta_lora.named_modules():
            if isinstance(module, LoRALinear):
                lora_params[name] = module.A @ module.B  # Compute AB^T

        for name, ab in lora_params.items():
            if torch.isnan(ab).any() or torch.isinf(ab).any():
                print(f"NaN or Inf detected in {name}")
                sys.exit(0)

        for name, ab in added_params.items():
            if torch.isnan(ab).any() or torch.isinf(ab).any():
                print(f"NaN or Inf detected in AddedParams {name}")
                sys.exit(0)


        loss1 = sum(torch.norm(ab.T - added_params[name], p=2) for name, ab in lora_params.items())

        if torch.isnan(loss1).any():
            print("⚠️ NaN detected in loss1!")
            sys.exit(0)

        return loss1

    def Phase2FairAttack(self, logits, targets, sensitive_labels, measure):


        loss_uf = DisparateImpactMaximizerLoss(measure=measure)
        loss2 = loss_uf(logits, targets, sensitive_labels)

        return loss2