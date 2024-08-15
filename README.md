A feature extractor that combines the subspaces of the observation space of different types and uses different layers of neural networks to extract features from each subspace. The layers include a simple flattening layer, an embedding layer, a CNN, and two different RNNs: GRU and LSTM with optional attention as in Transformer models. The last two are flexibly configurable, see the example. The features are then concatenated and passed through the main neural network defined for the deep reinforcement learning algorithm.

I also prepare a test environment to test the feature extractor with a simple out-of-the-box algorithm (PPO). The observation space of the test environment combines 4 data types: categorical, vector, image, and a variable-length sequence. The step performs a dummy update. The feature extractor is designed to handle these data types, extract features from them, and concatenate them.

## Install
```bash
pip install torch stable-baselines3 gymnasium numpy
```

## Usage
```python try.py```

## Author
Nikita Smirnov.
Please contact me in case of any questions or bug reports: [mailto](mailto:detectivecolombo@gmail.com)