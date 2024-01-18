import torch


class WorldModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model.to(self.device)

    def observe(self, image, latent, action):
        pass

    def train(self, replay_buffer):
        pass
