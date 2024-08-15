import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Dict, Box


class MultiObsDummyEnv(gymnasium.Env):
    FEATURE_DIM = 3
    MAX_SEQUENCE_LENGTH = 10

    def __init__(self, episode_length=100):
        self.episode_length = episode_length

        self.observation_space = Dict(
            {
                # utility space to indicate episode has done and RNN state need to be resetted afterwards
                "_episode_starts": Discrete(2),
                # discrete space with 5 possible values
                "dis_ind": Discrete(5),
                # 1-dim vector
                "vec_pos": Box(low=0, high=1, shape=(10,), dtype=np.float32),
                # 3-dim image space
                "img_grid": Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
                # 2-dim Box imitating a sequence (unfortunately, there is no gymnasium.Sequence support in sb3 yet)
                "seq_responses": Box(
                    low=0.0, high=1.0, shape=(self.MAX_SEQUENCE_LENGTH, self.FEATURE_DIM), dtype=np.float32
                ),
            }
        )

        self.action_space = Discrete(3)

    def step(self, action):
        self.steps += 1
        self.state["_episode_starts"] = 0

        self.state["dis_ind"] = self.observation_space["dis_ind"].sample()
        self.state["vec_pos"] = self.observation_space["vec_pos"].sample()
        self.state["img_grid"] = self.observation_space["img_grid"].sample()

        # dummy updates
        self.state["vec_pos"] += np.random.uniform(-0.1, 0.1)
        # randomly change some pixels in the image based on the action
        num_pixels_to_change = np.random.randint(0, 100)
        for _ in range(num_pixels_to_change):
            channel = np.random.randint(0, 3)
            x = np.random.randint(0, 64)
            y = np.random.randint(0, 64)
            self.state["img_grid"][channel, x, y] = np.random.randint(0, 256)
        # randomly generate a sequence with padding
        self.state["seq_responses"] = np.zeros(self.observation_space["seq_responses"].shape, dtype=np.float32)
        seq_length = np.random.randint(1, self.MAX_SEQUENCE_LENGTH + 1)
        print(f"Sequence Length: {seq_length}")
        self.state["seq_responses"][:seq_length, :] = self.observation_space["seq_responses"].sample()[:seq_length, :]

        reward = (
            self.state["dis_ind"]
            + np.mean(self.state["vec_pos"])
            + np.sum(self.state["img_grid"][0]) / 255.0
            + np.sum(self.state["seq_responses"][0])
        )

        terminated = False
        truncated = self.steps >= self.episode_length

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.state = {
            "_episode_starts": 1,
            "dis_ind": 0,
            "vec_pos": np.zeros(self.observation_space["vec_pos"].shape, dtype=np.float32),
            "img_grid": np.zeros(self.observation_space["img_grid"].shape, dtype=np.uint8),
            "seq_responses": np.zeros(self.observation_space["seq_responses"].shape, dtype=np.float32),
        }
        self.steps = 0
        return self.state, {}
