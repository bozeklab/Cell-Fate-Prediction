import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import Adam
from torchmetrics.classification import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)

import wandb


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data.

    This class implements positional encoding as described in the Transformer architecture. It adds a unique positional information to each position in the input sequence, which helps the model to understand the position of each element in the sequence.

    Args:
        num_hiddens (int): Dimensionality of the encoding.
        dropout (float): Dropout rate.
        max_len (int, optional): Maximum length of the input sequences. Default is 200.
    """

    def __init__(self, num_hiddens, dropout, max_len=200):
        """
        Initializes the PositionalEncoding layer.

        Args:
            num_hiddens (int): Dimensionality of the encoding.
            dropout (float): Dropout rate.
            max_len (int, optional): Maximum length of the input sequences. Default is 200.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len + 1  # Add cls token
        self.P = torch.zeros((1, self.max_len, num_hiddens))
        X = torch.arange(self.max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """
        Forward pass for adding positional encoding to the input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape (N, T, D), where N is the batch size, T is the sequence length,
                              and D is the dimensionality of each element in the sequence.

        Returns:
            torch.Tensor: Output tensor with positional encoding added, of shape (N, T, D).
        """
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class VideoModel(L.LightningModule):
    def __init__(
        self,
        backbone,
        batch_size,
        learning_rate,
        num_channels,
        nhead,
        nlayers,
        hidden_dim,
        dropout,
        pos_dropout,
        weight_decay,
        max_len=200,
        save_attention_weights=False,
    ):
        """
        Video classification model for cell fate prediction using a CNN backbone
        followed by a Transformer encoder over temporal features.

        The model extracts spatial features from each frame using a backbone CNN,
        models temporal dependencies with a Transformer encoder, and predicts
        a binary outcome (e.g., apoptosis vs. mitosis).

        Architecture:
            Frames → CNN Backbone → Feature Sequence →
            Positional Encoding → Transformer Encoder →
            Final Attention Pooling → Classification Head

        Args:
            backbone (torch.nn.Module):
                CNN feature extractor applied independently to each frame.
                Must output a feature vector of size `num_channels`.

            batch_size (int):
                Batch size used during training. Needed for Lightning logging.

            learning_rate (float):
                Learning rate for the optimizer.

            num_channels (int):
                Dimensionality of the per-frame feature embeddings and
                Transformer hidden size.

            nhead (int):
                Number of attention heads in the Transformer encoder layers.

            nlayers (int):
                Number of stacked Transformer encoder layers.

            hidden_dim (int):
                Hidden dimension of the feedforward network inside each
                Transformer encoder layer.

            dropout (float):
                Dropout probability used in Transformer layers.

            pos_dropout (float):
                Dropout applied after positional encoding.

            weight_decay (float):
                Weight decay (L2 regularization) applied in the optimizer.

            max_len (int, optional):
                Maximum sequence length supported by positional encoding.
                Default is 200.

            save_attention_weights (bool, optional):
                If True, stores attention weights from the final attention
                layer for interpretability or visualization.
        """
        super(VideoModel, self).__init__()
        self.backbone = backbone
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.loss_func = nn.BCEWithLogitsLoss()

        self.train_dying_cells = 0
        self.train_dividing_cells = 0

        self.val_dying_cells = 0
        self.val_dividing_cells = 0

        self.metrics = self.initialize_metrics()

        self.pos_enc = PositionalEncoding(
            num_channels, dropout=pos_dropout, max_len=max_len
        )
        encoder_layers = TransformerEncoderLayer(
            num_channels, nhead, hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        if save_attention_weights:
            for layer in self.transformer_encoder.layers:
                attention_wrapper(layer.self_attn)

        self.norm_layer_1 = nn.LayerNorm(num_channels)

        self.final_attn = nn.MultiheadAttention(
            num_channels, 1, dropout=dropout, batch_first=True
        )
        if save_attention_weights:
            self.attention_weights = None
            attention_wrapper(self.final_attn)
            self.final_attn.register_forward_hook(self.hook_fn)

        self.norm_layer_2 = nn.LayerNorm(num_channels)

        self.fc = nn.Linear(num_channels, 1)
        self.class_token = nn.Parameter(torch.zeros(1, 1, num_channels))

    def forward(self, x, src_mask=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor):
                Input tensor of shape (N, T, C, H, W) where:
                    N = batch size
                    T = number of time steps (frames)
                    C = number of channels
                    H = image height
                    W = image width

            src_mask (torch.Tensor, optional):
                Boolean padding mask of shape (N, T+1) used by the Transformer
                to ignore padded timesteps. The +1 accounts for the class token.

        Returns:
            torch.Tensor:
                Logits of shape (N, 1) for binary classification.
        """
        batch_size, timesteps, C, H, W = x.size()

        x = x.view(batch_size * timesteps, C, H, W)
        x = self.backbone(x)
        x = x.view(batch_size, timesteps, -1)

        class_tokens = self.class_token.expand(batch_size, 1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        x = self.pos_enc(x)

        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)

        x = self.norm_layer_1(x)
        x = self.final_attn(x, x, x, key_padding_mask=src_mask)[0]
        x = self.norm_layer_2(x)

        x = x[:, 0]
        x = self.fc(x)
        return x

    def _step(self, batch, mode):
        """
        Shared step used for training, validation, and testing.

        Performs:
            - Forward pass
            - Loss computation
            - Prediction thresholding
            - Metric updates
            - Class counting

        Args:
            batch (tuple):
                Batch returned by the dataloader containing:
                    cell_sequence (Tensor): frame sequence
                    mask (Tensor): padding mask
                    labels (Tensor): ground-truth labels
                    *_: unused metadata

            mode (str):
                One of {"Train", "Val", "Test"} indicating the current stage.

        Returns:
            torch.Tensor:
                Computed loss for the batch.
        """

        cell_sequence, mask, labels, _, _ = batch

        preds = self(cell_sequence, mask)

        labels = labels.float().unsqueeze(1)

        loss = self.loss_func(preds, labels)

        predicted_labels = (torch.sigmoid(preds) >= 0.5).float()

        self._update_metrics(mode, predicted_labels, labels)

        return loss

    def _log(self, loss, mode):
        """
        Logs loss values using Lightning's logging system.

        Args:
            loss (torch.Tensor):
                Loss value for the current batch.

            mode (str):
                Training stage ("Train", "Val", or "Test").
        """

        self.log(
            f"{mode}/Loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def _update_metrics(self, mode, predicted_labels, labels):
        """
        Updates all evaluation metrics for a given mode.

        Args:
            mode (str):
                One of {"Train", "Val", "Test"}.

            predicted_labels (torch.Tensor):
                Binary predictions after thresholding.

            labels (torch.Tensor):
                Ground truth labels.
        """

        predicted_labels = predicted_labels.cpu()
        labels = labels.cpu()
        for _, metric in self.metrics[mode].items():
            metric.update(predicted_labels, labels)

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min"),
            "monitor": "Val/Loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "Train")

        self._log(loss, "Train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "Val")

        self._log(loss, "Val")

        return loss

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
        loss = self._step(batch, "Test")

        self._log(loss, "Test")

        return loss

    def on_epoch_end(self, mode):
        """
        Computes and logs epoch-level metrics.

        This includes:
            - Accuracy
            - Precision
            - Recall
            - F1-score
            - Confusion matrix

        During testing, a normalized confusion matrix visualization is saved
        and optionally logged to Weights & Biases.

        Args:
            mode (str):
                One of {"Train", "Val", "Test"}.
        """

        for metric_name, metric in self.metrics[mode].items():
            if metric_name == "confusion_matrix":
                continue
            metric_results = metric.compute()
            if metric.average == "macro":
                self.log(f"{mode}/{metric_name.capitalize()}", metric_results)
            else:
                for i in range(2):
                    label = "Apoptosis" if i == 0 else "Mitosis"
                    self.log(
                        f"{mode}/{metric_name.capitalize()}_{label}", metric_results[i]
                    )

            metric.reset()

        confusion_matrix = self.metrics[mode]["confusion_matrix"].compute().numpy()

        if mode == "Test":
            # create a single confusion matrix
            normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
            annot = np.empty_like(normalized).astype(str)
            for i in range(normalized.shape[0]):
                for j in range(normalized.shape[1]):
                    annot[i, j] = (
                        f"{normalized[i, j]:.3f}\n({int(confusion_matrix[i, j])})"
                    )

            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(
                normalized,
                annot=annot,
                fmt="",
                cmap="Blues",
                xticklabels=["Apoptosis", "Mitosis"],
                yticklabels=["Apoptosis", "Mitosis"],
                annot_kws={"size": 14},
            )
            ax.set_xlabel("Predicted", fontsize=14)
            ax.set_ylabel("True", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.set_title("Confusion Matrix", fontsize=16)

            ax.collections[0].colorbar.ax.tick_params(labelsize=12)
            ax.collections[0].colorbar.ax.yaxis.label.set_size(14)

            eval_dir = os.path.join(
                self.trainer.checkpoint_callback.dirpath.split("training")[0],
                "evaluation",
            )

            os.makedirs(eval_dir, exist_ok=True)

            for post_fix in ["eps", "png"]:
                output_file = os.path.join(
                    self.trainer.default_root_dir, f"confusion_matrix.{post_fix}"
                )
                plt.savefig(output_file)

            plt.close()

            if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                wandb.log({"Confusion Matrix": wandb.Image(output_file)})

        self.metrics[mode]["confusion_matrix"].reset()

        if mode == "Train":
            if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                self.logger.log_metrics({f"{mode}/DyingCells": self.train_dying_cells})
                self.logger.log_metrics(
                    {f"{mode}/DividingCells": self.train_dividing_cells}
                )

            self.train_dying_cells = 0
            self.train_dividing_cells = 0
        elif mode == "Val":
            if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                self.logger.log_metrics({f"{mode}/DyingCells": self.val_dying_cells})
                self.logger.log_metrics(
                    {f"{mode}/DividingCells": self.val_dividing_cells}
                )

            self.val_dying_cells = 0
            self.val_dividing_cells = 0

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end("Train")

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end("Val")

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end("Test")

    def initialize_metrics(self):
        """
        Initializes metric collections for training, validation, and testing.

        Metrics include:
            - Accuracy
            - Precision
            - Recall
            - F1-score
            - Confusion matrix

        Metrics are tracked both as:
            - macro averages
            - per-class values

        Returns:
            dict:
                Nested dictionary containing metric instances per mode.
        """
        metrics = {}
        modes = ["Train", "Val", "Test"]
        metric_classes = [Accuracy, Precision, Recall, F1Score]
        averages = ["macro", None]

        for mode in modes:
            metrics[mode] = {}
            for metric_class in metric_classes:
                for average in averages:
                    if average is None and isinstance(metric_class, F1Score):
                        continue
                    elif average is None:
                        metric_name = f"{metric_class.__name__.lower()}_per_class"
                    else:
                        metric_name = metric_class.__name__.lower()
                    metrics[mode][metric_name] = metric_class(
                        task="multiclass", average=average, num_classes=2
                    )
            metrics[mode]["confusion_matrix"] = ConfusionMatrix(
                task="multiclass", num_classes=2
            )

        return metrics

    def hook_fn(self, module, input, output):
        """
        Forward hook used to capture attention weights from the final
        multi-head attention layer.

        Args:
            module (nn.Module):
                The module to which the hook is attached.

            input (tuple):
                Inputs passed to the module.

            output (tuple):
                Module output containing both values and attention weights.
        """
        self.attention_weights = output[1]

    def get_attention_weights(self):
        """
        Returns normalized attention weights from the final attention layer.

        The attention weights correspond to the importance of each frame
        in the sequence relative to the class token.

        Returns:
            torch.Tensor:
                Normalized attention weights for each timestep.

        Raises:
            ValueError:
                If attention weights were not stored during the forward pass.
        """
        if self.attention_weights is None:
            raise ValueError(
                "Attention weights not available. Ensure save_attention_weights=True."
            )
        self.attention_weights = self.attention_weights[0, 0, 0, 1:]
        # normalize attention weights
        attn_min = torch.min(self.attention_weights)
        attn_max = torch.max(self.attention_weights)
        attention_weights = (self.attention_weights - attn_min) / (attn_max - attn_min)

        return attention_weights


def attention_wrapper(module):
    """
    Wraps a MultiheadAttention module to ensure attention weights are returned
    without averaging across heads.

    This allows inspection of individual attention head contributions.

    Args:
        module (torch.nn.Module):
            MultiheadAttention module to be modified.
    """

    forward_orig = module.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    module.forward = wrap
