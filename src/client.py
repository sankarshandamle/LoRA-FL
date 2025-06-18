import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.fair_measures import compute_equalized_odds, compute_demographic_parity, compute_equal_opportunity


class Device:
    def __init__(self, id, dataset, train_indices, test_indices, raw_frame, HONEST_LR, ADV_LR, Net, LoRANet, CustomTwoPhaseLowRankLoss, sensitive_feature = 'sex', adversarial = False, iid=True, rank = 4) -> None:

        input_size = len(dataset[0][0])
        self.id = id

        self.model = Net(input_size=input_size)
        self.global_model = Net(input_size=input_size)
        # self.model.to(torch.device("cuda"))

        if(adversarial):
            self.lora_model = LoRANet(input_size=input_size, rank=rank).cuda()
            self.criterion_honest = nn.BCEWithLogitsLoss()
            self.criterion_adv = CustomTwoPhaseLowRankLoss() #
            self.optimizer_lora_phase1 = torch.optim.AdamW(
                                        [p for p in self.lora_model.parameters() if p.requires_grad], lr=ADV_LR
                                  )
            # first freeze A adapter
            self.freezeAadapter()
            self.optimizer_lora_phase2 = torch.optim.AdamW(
                                        [p for p in self.lora_model.parameters() if p.requires_grad], lr=ADV_LR
                                  )
            # unfreeze A adapter
            self.unfreezeAadapter()

        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=HONEST_LR)

        self.train_indices = train_indices
        self.test_indices = test_indices

        self.train_set = Subset(dataset, train_indices)
        self.test_set = Subset(dataset, test_indices)

        self.sensitive_feature = sensitive_feature
        self.raw_frame = raw_frame

        self.adversarial = adversarial

        self.train_loader = DataLoader(self.train_set, batch_size=2048, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=512, shuffle=True)
        self.iid = iid


    def train(self, train_loader = None, num_epochs = 5 ):
        for device_epoch in range(num_epochs):
            self.model.train()
            losses = []
            for inputs, labels, indices in self.train_loader:
                self.optimizer.zero_grad()

                # inputs, train_labels = inputs.cuda(), train_labels.cuda()  # add this line
                labels = labels.float()
                # print(indices)

                sensitive_label = self.raw_frame.loc[indices.tolist(), self.sensitive_feature].tolist()

                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                labels = labels.view_as(outputs)
                if(self.adversarial):
                    loss = self.criterion_honest(outputs, labels)
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            # print(f"loss: {np.array(losses).mean()}")

    def train_adversarial(self, theta_g, theta_i, train_loader = None, num_epochs = 2, measure = 'eo'):

        for device_epoch in range(num_epochs):

            self.lora_model.train()
            losses_phase1 = []
            losses_phase2 = []

            for inputs, labels, indices in self.train_loader:

                self.optimizer_lora_phase1.zero_grad()
                self.optimizer_lora_phase2.zero_grad()

                """
                Phase 1 : Regularizer
                """

                # inputs, train_labels = inputs.cuda(), train_labels.cuda()  # add this line
                labels = labels.float()
                # print(indices)

                sensitive_label = self.raw_frame.loc[indices.tolist(), self.sensitive_feature].tolist()

                outputs = self.lora_model(inputs)
                outputs = outputs.squeeze()
                labels = labels.view_as(outputs)

                if torch.isnan(outputs).any():
                    print("⚠️ NaN detected in logits!")
                    sys.exit(0)

                loss_phase1 = self.criterion_adv.Phase1Regularizer(theta_g, theta_i, self.lora_model, outputs, labels, sensitive_label)
                if torch.isnan(loss_phase1).any():
                    print("⚠️ NaN detected in loss!")
                    sys.exit(0)


                loss_phase1.backward()
                self.optimizer_lora_phase1.step()
                losses_phase1.append(loss_phase1.item())

                """
                Phase 2: Fairness Attack
                """
                outputs = self.lora_model(inputs)
                outputs = outputs.squeeze()
                labels = labels.view_as(outputs)

                loss_phase2 = self.criterion_adv.Phase2FairAttack(outputs, labels, sensitive_label, measure)
                loss_phase2.backward()
                self.optimizer_lora_phase2.step()
                losses_phase2.append(loss_phase2.item())
                if torch.isnan(loss_phase2).any():
                    print("⚠️ NaN detected in loss!")
                    sys.exit(0)


    def validate(self, test_loader=None, flag=False, verbose=False):
        # Set the model to evaluation mode
        self.model.eval()

        # Use the default test_loader if none is provided
        if test_loader is None:
            test_loader = self.test_loader

        total = 0
        correct = 0
        losses = []

        pr_y1_unpriv = 0
        pr_y1_priv = 0
        total_unpriv = 0
        total_priv = 0

        y_true_priv, y_pred_priv = [], []
        y_true_unpriv, y_pred_unpriv = [], []

        preds_priv, preds_unpriv = [], []  # Needed for Demographic Parity


        with torch.no_grad():
            for inputs, labels, indices in self.test_loader:

                # Move data to GPU if needed (uncomment if using CUDA)
                # inputs, labels = inputs.cuda(), labels.cuda()

                # Convert labels to float for BCEWithLogitsLoss
                labels = labels.float()

                sensitive_label = self.raw_frame.loc[indices.tolist(), self.sensitive_feature].tolist()


                # Forward pass
                if self.adversarial and flag:
                    outputs = self.lora_model(inputs)
                else:
                    outputs = self.model(inputs)

                # Adjust shapes
                outputs = outputs.squeeze()
                labels = labels.view_as(outputs)

                # Compute loss
                if(self.adversarial):
                    loss = self.criterion_honest(outputs, labels)
                else:
                    loss = self.criterion(outputs, labels)
                losses.append(loss.item())

                # Apply sigmoid to logits and threshold at 0.5 for predictions
                predictions = torch.sigmoid(outputs) >= 0.5

                for i in range(len(predictions)):
                    if sensitive_label[i] == 1:  #  Privileged group "Male" mapped to "1" for Adult
                        total_priv += 1
                        if predictions[i] == labels[i]:
                            pr_y1_priv += 1
                        y_true_priv.append(labels[i].item())
                        y_pred_priv.append(predictions[i].item())
                        preds_priv.append(predictions[i].item())


                    else:
                        total_unpriv += 1
                        if predictions[i] == labels[i]:
                            pr_y1_unpriv += 1
                        y_true_unpriv.append(labels[i].item())
                        y_pred_unpriv.append(predictions[i].item())

                        preds_unpriv.append(predictions[i].item())

                # Update total and correct counts
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        # disparate_impact = pr_y1_unpriv/total_unpriv / (pr_y1_priv/total_priv)
        equalized_odds = compute_equalized_odds(y_true_priv, y_pred_priv, y_true_unpriv, y_pred_unpriv)
        equalized_odds_pp = compute_equal_opportunity(y_true_priv, y_pred_priv, y_true_unpriv, y_pred_unpriv)
        demographic_parity = compute_demographic_parity(preds_priv, preds_unpriv)


        # Verbose logging
        if verbose:
            # print(f"Disparate Impact: {disparate_impact:.2f}")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Average Loss: {sum(losses) / len(losses):.4f}")

        return sum(losses) / len(losses), accuracy, equalized_odds, equalized_odds_pp, demographic_parity


    def get_model_params(self):
        # if self.adversarial:
        #     return convert_lora_to_standard(self.lora_model.state_dict(), 1.0)
        return self.model.state_dict()

    def set_global_model(self, current_global_model):
        self.global_model.load_state_dict(current_global_model)
        # print(missing, unexpected)

    def update_model(self, new_state_dict):
        self.model.load_state_dict(new_state_dict, strict=False)
        # print(missing, unexpected)

    def transfer_mlp_to_lora(self, new_state_dict):
        """
        Transfers the weights from a standard MLP model to the LoRANet model.

        - Copies fc1, fc2, fc3 weights from the MLP to `fc1.base_layer`, `fc2.base_layer`, and `fc3.base_layer` in LoRANet.
        - Leaves LoRA adapter parameters (A, B) unchanged.

        Args:
            lora_model (LoRANet): The LoRA model to be updated.
            mlp_model (nn.Module): The standard MLP model with matching architecture.
        """
        lora_state_dict = self.lora_model.state_dict()

        for name, param in new_state_dict.items():
            if "fc" in name and "weight" in name:  # Only transfer fully connected layers
                lora_name = f"{name.split('.')[0]}.base_layer.{name.split('.')[1]}"
                # print(lora_name)
                lora_state_dict[lora_name] = param  # Copy the MLP weights

            elif "bn" in name:  # Transfer batch normalization layers
                lora_state_dict[name] = param  # Copy BN weights

        # Load the modified state dict into the LoRA model
        missing, unexpected = self.lora_model.load_state_dict(lora_state_dict, strict=True)
        # print(missing, unexpected)

    def convert_lora_to_standard(self, alpha):
        """
        Converts the LoRA state dict to a standard MLP state dict by merging the base weights and the low-rank adaptations.

        Args:
            lora_state_dict (dict): The state dictionary from a LoRA-based model.
            alpha (float): The scaling factor for the LoRA low-rank adaptation.

        Returns:
            dict: The converted standard state dictionary.
        """
        # Initialize a dictionary to hold the converted state_dict
        standard_state_dict = {}
        lora_state_dict = deepcopy(self.lora_model.state_dict())

        # Convert each LoRA layer to a standard MLP layer by merging A @ B^T with base weights
        # For the LoRALinear layers (fc1, fc2, fc3)
        for layer in ['fc1', 'fc2', 'fc3']:
            # Compute the standard layer weight: theta' = theta + A @ B^T
            standard_state_dict[f"{layer}.weight"] = (
                lora_state_dict[f"{layer}.base_layer.weight"] +
                (alpha * lora_state_dict[f"{layer}.A"] @ lora_state_dict[f"{layer}.B"]).T
            )


        # Handle batch normalization layers
        for i in range(1, 3):
            standard_state_dict[f"bn{i}.weight"] = lora_state_dict[f"bn{i}.weight"]
            standard_state_dict[f"bn{i}.bias"] = lora_state_dict[f"bn{i}.bias"]
            standard_state_dict[f"bn{i}.running_mean"] = lora_state_dict[f"bn{i}.running_mean"]
            standard_state_dict[f"bn{i}.running_var"] = lora_state_dict[f"bn{i}.running_var"]
            standard_state_dict[f"bn{i}.num_batches_tracked"] = lora_state_dict[f"bn{i}.num_batches_tracked"]

        return standard_state_dict


    def add_adapters(self):
        # temp_model = deepcopy(self.model)
        self.model.load_state_dict(self.convert_lora_to_standard(1.0), strict=True)
        # total_norm = sum(torch.norm(param1 - param2, p=2) ** 2
                    #  for (name, param1), (_, param2) in zip(temp_model.named_parameters(), self.model.named_parameters()))
        # print(f"Total norm: {total_norm}")


    def freezeAadapter(self):

        for name, param in self.lora_model.state_dict().items():
            if 'A' in name:  # Look for A adapters
                param.requires_grad = False  # Freeze the parameter

            if 'B' in name:
                param.requires_grad = True

    def unfreezeAadapter(self):

        for name, param in self.lora_model.state_dict().items():
            if 'B' in name:  # Look for B adapters
                param.requires_grad = True  # Freeze the parameter

            if 'A' in name:
                param.requires_grad = True
