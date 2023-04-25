"""
Federated learning server for the UNSW-NB15 task.
"""

import typing
import flwr as fl


def weighted_average(
    metrics: typing.List[typing.Tuple[int, typing.Dict[str, float]]]
) -> typing.Dict[str, float]:
    """
    Perform the weighted average of the flower evaluation metrics, using dataset size as the weighting

    Arguments:
    - metrics: flower evaluation metrics, a list with a tuple for each client
               that contains the dataset size and the dictionary of local metrics
    """
    total_examples = 0
    federated_metrics = {k: 0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    return {k: v / total_examples for k, v in federated_metrics.items()}


def create_learning_strategy() -> fl.server.strategy.Strategy:
    """
    Create the learning strategy to be followed by the server
    """
    return fl.server.strategy.FedAvg(
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=create_learning_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
