# HabitatPeer
## Introduction
This project aims to address the dependency conflict between the Habitat 0.3 and the agent. To achieve this, the project utilizes WebRTC technology through the `aiortc` library, enabling real-time communication between the habitat and the agent through DataChannel.
## Installation
To install the HabitatPeer, follow the steps below:
1. Clone the repository
2. Install the habitat according to [here](https://github.com/facebookresearch/habitat-lab)
3. Install the required packages using the command `pip install -r requirements.txt`
4. Install the dataset, the default dataset for this project is hm3d-minival-habitat-v0.2
## Usage
Follow these steps to run the system:
1. Start the signaling by running the command `python launch_server.py`
2. Start the callee peer, one of the available peer can be installed [here](https://github.com/leidZhang/SG-Nav)
3. Start the caller peer by running the command `python launch_env.py`

Note: You can modify the ip configs in the `ip_configs.json`, you can also modify the yaml file in the `config` folder to change the scene.
