import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from feature_extractor import MultiCombinedExtractor
from env import MultiObsDummyEnv

LAYERS_CONFIG = {
    "cnn": {
        "cnn_output_size": 256,  # final output size [will be feed into F.relu(nn.Linear(cnn_flattened_output_dim, cnn_output_size))]
        "normalized_image": False,  # if image is already normalized (see sb3's internal implementation of NatureCNN)
    },
    "rnn": {
        "hidden_state_size": 64,  # output size of the recurrent layer
        "num_layers": 2,  # number of hidden layers
        "layer_type": "lstm",  # lstm or gru
        "dropout": 0.0,  # dropout rate
        "bidirectional": True,  # whether to use bidirectional RNN (l-r and r-l sequence processing)
        "attention": "none",  # none, dot, general
    },
}


def main():
    env = DummyVecEnv([lambda: MultiObsDummyEnv(episode_length=5)])

    policy_kwargs = dict(
        features_extractor_class=MultiCombinedExtractor,
        features_extractor_kwargs=dict(
            layers_config=LAYERS_CONFIG,
            device=th.device("cpu"),
        ),
    )

    # n_steps = 4 just to check that gradient update works too
    model = PPO("MultiInputPolicy", env, n_steps=4, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(10)


if __name__ == "__main__":
    main()
