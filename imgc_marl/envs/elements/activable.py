from simple_playgrounds.common.texture import ColorTexture
from simple_playgrounds.element.elements.activable import RewardOnActivation


class CustomRewardOnActivation(RewardOnActivation):
    def __init__(self, terminate: bool, **kwargs):
        super().__init__(**kwargs)
        self.change_state(terminate)

    @property
    def terminate_upon_activation(self):
        return self.terminate

    def activate(self, _):
        self.activated = True
        return None, None

    def deactivate(self):
        self.activated = False

    def change_state(self, terminate: bool):
        """Makes the object activable and reward-providing or the contrary"""
        self.reward = int(terminate)
        self.terminate = terminate
        # Set different color when the object is the activable one
        if terminate:
            self._texture_surface.fill(color=(255, 0, 0))
        else:
            self._texture_surface.fill(color=(0, 255, 0))
