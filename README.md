# Machine Learning and Federated Learning Techniques for Anomaly Detection

A presentation I did on the step-by-step process for machine learning and federated learning of an Intrusion Detection System (IDS).


## Prerequisites

To run the code here, you will need to download the `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` files from https://research.unsw.edu.au/projects/unsw-nb15-dataset
and place them in a `data/` folder in this root.

Then you will need to install the python library from the requirements file:

```sh
pip install -r requirements.txt
```

## Running the code

The presentation notebook for the data preprocessing, supervised learning, and unsupervised
can be simply opened using jupyter notebook or jupyter lab.

The federated learning portion requires three terminals, first start the server with
`python server.py`, then start at least two clients with `python client.py`. The simulation
version requires installation of the flower simulation library with `pip install flwr[simulation]`, and then can be run with `python simulation.py`
