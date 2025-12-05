# server.py
import flwr as fl
from flwr.server.strategy import FedAvg
import argparse

def main(rounds: int = 10, port: int = 8080):
    strategy = FedAvg(
        fraction_fit=1.0,      # use all clients (cross-silo)
        min_fit_clients=4,
        min_eval_clients=4,
        min_available_clients=4,
    )
    print(f"Starting Flower server on 0.0.0.0:{port}, rounds={rounds}")
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config={"num_rounds": rounds},
        strategy=strategy
    )

if __name__ == "__main__":
    import fire
    fire.Fire(main)
