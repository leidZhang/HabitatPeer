# HabitatPeer
## Usage
To use HabitatPeer, navigate to the project folder, then follow these steps:
1. Make sure you can connect to the signaling server, you can either use a public server or run your own by using `python launch_server.py`.
2. Start the receiver peer by running `python launch_receiver.py`, you can also run it on another machine if you want to receive the stream from another device, but make sure to specify the stun servers.
3. Start the provider peer by running `python launch_provider.py`, similar to starting the receiver peer, you can also run it on another machine if you want to provide the stream to another device, but make sure to specify the stun servers.

If you want to specify the ip settings, you can change the values in `ip_configs.json`