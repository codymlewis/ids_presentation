"""
Simulate the federated learning of the UNSW-NB15 dataset
"""


import flwr as fl

import server
import client


def create_client(_cid) -> client.Client:
    """
    Create a client for the simulation

    Arguments:
    - _cid: Placeholder for the client id
    """
    return client.Client()


if __name__ == "__main__":
    history = fl.simulation.start_simulation(
        client_fn=create_client,
        num_clients=2,
        strategy=server.create_learning_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
