from abc import ABC, abstractmethod

import torch


class BaseRewardLoss(ABC):
    """
    Base class for reward functions implementing a differentiable reward function for optimization.
    """

    def __init__(self, name: str, weighting: float):
        self.name = name
        self.weighting = weighting

    @staticmethod
    def freeze_parameters(params: torch.nn.ParameterList):
        for param in params:
            param.requires_grad = False

    @abstractmethod
    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_text_features(self, prompt: str | list[str]) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        pass

    def process_features(self, features: torch.Tensor) -> torch.Tensor:
        features_normed = (features / features.norm(dim=-1, keepdim=True)).to(features.dtype)
        return features_normed

    def __call__(self, image: torch.Tensor, prompt: str | list[str], raw_image: torch.Tensor) -> torch.Tensor:
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(prompt)

        image_features_normed = self.process_features(image_features)
        text_features_normed = self.process_features(text_features)

        loss = self.compute_loss(image_features_normed, text_features_normed)
        return loss


# class RedRewardLoss:
#     """
#     A simple reward that measures redness of an image.
#     """
#     def __init__(self, weighting: float):
#         self.name = "Redness"
#         self.weighting = weighting

#     def __call__(self, image: torch.Tensor, prompt: str, raw_image: torch.Tensor) -> torch.Tensor:
#         # Assuming image is in the format (B, C, H, W)
#         red_channel = raw_image[:, 0, :, :]
#         green_channel = raw_image[:, 1, :, :]
#         blue_channel = raw_image[:, 2, :, :]

#         # Calculate redness as a simple ratio of red to green and blue channels
#         redness = (green_channel + blue_channel) / 2 - red_channel
#         return redness.mean()


class RedRewardLoss:
    """
    A simple reward that measures redness of an image.
    """
    def __init__(self, weighting: float):
        self.name = "Redness"
        self.weighting = weighting

    def __call__(self, image: torch.Tensor, prompt: str, raw_image: torch.Tensor) -> torch.Tensor:
        # Assuming image is in the format (B, C, H, W)
        red_channel = raw_image[:, 0, :, :]
        green_channel = raw_image[:, 1, :, :]
        blue_channel = raw_image[:, 2, :, :]

        # Calculate redness as a simple ratio of red to green and blue channels
        redness = (green_channel + blue_channel) / 2 - red_channel
        return redness.mean()