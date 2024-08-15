import gymnasium
import numpy as np
import torch as th
from stable_baselines3.common.policies import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.utils import zip_strict
from torch import nn as nn
from typing import Dict

from rnn_attention import RnnAttention


class MultiCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces with image/sequence/vector/discrete subspaces.
    Supports Embedding for discrete subspaces, CNN for image subspaces, RNN for sequence subspaces, configurated via `layers_config`.
    NOTE: subspaces' names should start with a typekey to be recognized by the extractor, e.g., "dis_bla", "img_blub", "seq_blib", "vec_blob"
    NOTE: observation should contain an utility subspace "_episode_starts" of type Discrete(2), which indicates if an episode has started
    NOTE: sequences assumed to be padded by the environment to fit into the Box shape (max_sequence_length, feature_dim)

    :param observation_space: expected Dict observation space with mixed-type subspaces
    :param layers_config: configuration of the layers
    :param variadic_treatment: how to treat variadic sequences, options: "pack" (unpad and pack), "pad" (feed padded), default: "pack"
    :param device: device, default: "cpu"
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        layers_config: Dict[str, Dict[str, int | bool | str]],
        device: th.device = th.device("cpu"),
    ) -> None:
        # to initialize the base class set features_dim to 1, at the end it is manually reassigned to the resulting flattened dim
        super().__init__(observation_space, features_dim=1)

        self.layers_config = layers_config
        self.device = device

        self.rnn_state = None
        self.rnn_batch_size = 1
        self.rnn_num_directions = 1

        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if "dis" in key and isinstance(subspace, gymnasium.spaces.Discrete):
                # should be already one-hot encoded by sb3
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
            elif "vec" in key and isinstance(subspace, gymnasium.spaces.Box):
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
            elif "img" in key and is_image_space(subspace, self.layers_config["cnn"]["normalized_image"]):
                cnn_output_size = self.layers_config["cnn"]["cnn_output_size"]
                extractors[key] = NatureCNN(
                    subspace,
                    features_dim=cnn_output_size,
                    normalized_image=self.layers_config["cnn"]["normalized_image"],
                )
                total_concat_size += cnn_output_size
            elif "seq" in key and isinstance(subspace, gymnasium.spaces.Box):
                # 2-dim Box with a shape: (max_sequence_length, feature_dim)
                rnn_input_dim = subspace.shape[1]
                rnn_output_dim = self.layers_config["rnn"]["hidden_state_size"]
                # init RNN layer
                if self.layers_config["rnn"]["layer_type"] == "lstm":
                    extractors[key] = nn.LSTM(
                        rnn_input_dim,
                        rnn_output_dim,
                        self.layers_config["rnn"]["num_layers"],
                        dropout=self.layers_config["rnn"]["dropout"],
                        bidirectional=self.layers_config["rnn"]["bidirectional"],
                        batch_first=True,
                    )
                else:
                    extractors[key] = nn.GRU(
                        rnn_input_dim,
                        rnn_output_dim,
                        self.layers_config["rnn"]["num_layers"],
                        dropout=self.layers_config["rnn"]["dropout"],
                        bidirectional=self.layers_config["rnn"]["bidirectional"],
                        batch_first=True,
                    )

                # orthogonal initialization of weights
                for name, param in extractors[key].named_parameters():
                    if "bias" in name:
                        nn.init.constant_(param, 0)
                    elif "weight" in name:
                        nn.init.orthogonal_(param, np.sqrt(2))

                self.rnn_num_directions = 1 if not self.layers_config["rnn"]["bidirectional"] else 2
                total_concat_size += self.rnn_num_directions * rnn_output_dim

                # init Attention layer
                if self.layers_config["rnn"]["attention"] != "none":
                    self.rnn_attention = RnnAttention(
                        self.layers_config["rnn"]["attention"],
                        self.layers_config["rnn"]["hidden_state_size"] * self.rnn_num_directions,
                        self.device,
                    )
                else:
                    self.rnn_attention = None

        # NOTE: it is CRUCIAL to save all subextractors under a nn.ModuleDict,
        # otherwise a they won't be registered and weights won't be backpropagated
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        extractor_outputs = []
        for key, extractor in self.extractors.items():
            if key in observations:
                if "seq" in key:
                    extractor_output = self._process_sequences(
                        observations[key], observations["_episode_starts"], extractor
                    )
                else:
                    extractor_output = extractor(observations[key])

                extractor_outputs.append(extractor_output)

        if len(extractor_outputs) > 0:
            all_features = th.cat(extractor_outputs, dim=1)
        else:
            all_features = th.zeros(1, self._features_dim)

        return all_features

    def _process_sequences(
        self, sequence_tensor: th.Tensor, episode_starts: th.Tensor, rnn_layer: nn.LSTM | nn.GRU
    ) -> th.Tensor:
        """
        Processes a sequence tensor with a rnn layer. Based on sb3contrib's RecurrentActorCriticPolicy class method.

        :param sequence_tensor: tensor of shape (batch_size, padded_sequence_length, feature_dim) containing padded sequences
        :param episode_starts: tensor of shape (batch_size, 2) indicating if an episode has started (second dim is one-hot encoded binary value)
        :param rnn_layer: rnn layer (nn.LSTM or nn.GRU)
        :returns: rnn layer output
        """
        # pytorch rnn layers expect by default the batch size to be the second dimension, like
        # (sequence_length, batch_size, features_dim), but sb3 uses (batch_size, sequence_length, features_dim),
        # so that if rnn was initialized with batch_first=False, the first two dimensions have to be swapped
        # with .swapaxes(0, 1) and transposed back in output. Here RNN is always initialized with batch_first=True.

        # check if the RNN state is initialized, provide default batch size as a parameter
        if self.rnn_state is None:
            self._init_rnn(sequence_tensor.shape[0])

        # NOTE: batch_size = number of envs for data collection or = number of sequences for gradient update.
        batch_size = sequence_tensor.shape[0]

        # switch to binary labels
        if len(episode_starts.shape) == 2:
            episode_starts = episode_starts.unsqueeze(0)
        episode_starts = th.argmax(episode_starts, dim=2).float()

        # feed batch if no episode starts are found
        if th.all(episode_starts == 0.0):
            # input: (batch_size, seq_len, hidden_dim),
            # rnn_output: (rnn_batch_size, -1, hidden_dim), rnn_state: (num_layers * num_directions, rnn_batch_size, hidden_dim)
            sequence_tensor = sequence_tensor.reshape(self.rnn_batch_size, -1, rnn_layer.input_size)
            rnn_output, self.rnn_state = rnn_layer(sequence_tensor, self.rnn_state)
        else:
            rnn_output = []
            # iterate over the sequences and reset the rnn state when found 1 in episode_starts
            # NOTE: here we don't have to reshape as it is anyway iterated over the first dimension
            for sequence, episode_start in zip_strict(sequence_tensor, episode_starts):
                # add batch dimension again
                sequence = sequence.unsqueeze(0)
                if self.layers_config["rnn"]["layer_type"] == "lstm":
                    # catch where episode starts and reset the rnn state
                    hidden_output, self.rnn_state = rnn_layer(
                        sequence,
                        (
                            (1.0 - episode_start) * self.rnn_state[0],
                            (1.0 - episode_start) * self.rnn_state[1],
                        ),
                    )
                    rnn_output += [hidden_output]
                else:
                    hidden_output, self.rnn_state = rnn_layer(
                        sequence,
                        (1.0 - episode_start) * self.rnn_state[0],
                    )
                    rnn_output += [hidden_output]
            rnn_output = th.cat(rnn_output)

        # extract the last hidden state from the RNN output, since we do not need sequential output
        if self.layers_config["rnn"]["layer_type"] == "lstm":
            final_state = self.rnn_state[0].view(
                self.layers_config["rnn"]["num_layers"],
                self.rnn_num_directions,
                self.rnn_batch_size,
                self.layers_config["rnn"]["hidden_state_size"],
            )[-1]
        else:
            final_state = self.rnn_state.view(
                self.layers_config["rnn"]["num_layers"],
                self.rnn_num_directions,
                self.rnn_batch_size,
                self.layers_config["rnn"]["hidden_state_size"],
            )[-1]

        # if bidirectional then concatenate of l-r and r-l hidden states, else just take the last hidden state
        if self.rnn_num_directions == 1:
            final_state = final_state.squeeze(0)
        elif self.rnn_num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            final_state = th.cat((h_1, h_2), 1)

        # forward pass through the attention layer if applicable
        if self.rnn_attention is not None:
            final_output, _ = self.rnn_attention(rnn_output, final_state)
        else:
            final_output = final_state

        return final_output

    def _init_rnn(self, batch_size: int = 1) -> None:
        """
        Initializes the state for LSTM or GRU layer as well as attention layer if applicable.

        :param batch_size: initial batch_size, which will be used for all subsequent states
        """

        def _fill() -> th.Tensor:
            return th.zeros(
                (
                    self.layers_config["rnn"]["num_layers"] * self.rnn_num_directions,
                    batch_size,
                    self.layers_config["rnn"]["hidden_state_size"],
                ),
                dtype=th.float32,
                device=self.device,
            )

        self.rnn_batch_size = batch_size
        if self.layers_config["rnn"]["layer_type"] == "lstm":
            self.rnn_state = (_fill(), _fill())
        else:
            self.rnn_state = _fill()


"""
packing:
               # NOTE: if a sequence is fully padded with zeros, then it needs a special treatment to avoid zero sequence length for packing.
                # That could be in two cases: a) the initial state for a new episode, b) there were no observations for a sequence timestep.
                # It is therefore decided to clamp lengths tensor with min=1.0, so that for cases mentioned above, we have 1-size
                # sequence of feature-dim zeros, which shouldn't seriously affect the RNN and at the same time carries some sense with it.
                lengths = (sequence_tensor.sum(dim=-1) != 0).sum(dim=-1).clamp(min=1.0).int()
                packed_sequence_tensor = pack_padded_sequence(
                    sequence_tensor, lengths=lengths, batch_first=True, enforce_sorted=False
                )
                packed_rnn_output, self.rnn_state = rnn_layer(packed_sequence_tensor, self.rnn_state)
                unpacked_rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)
                return th.flatten(unpacked_rnn_output, start_dim=0, end_dim=1)
"""
