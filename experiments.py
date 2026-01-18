import torch
from banner import show_banner
from data import load_data
from scaling import get_scaler
from model import Net
from train import train


def ask_choice(title, options):
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  [{i}] {o}")
    while True:
        c = input(">>> ")
        if c.isdigit() and 1 <= int(c) <= len(options):
            return options[int(c) - 1]


def ask_int(msg, default):
    v = input(f"{msg} (default={default}): ")
    return int(v) if v.isdigit() else default


def ask_float(msg, default):
    v = input(f"{msg} (default={default}): ")
    try:
        return float(v)
    except:
        return default


def main():
    show_banner()

    scaler = ask_choice(
        "Choose data normalization:",
        ["none", "minmax", "standard", "robust", "l2"]
    )

    activation = ask_choice(
        "Choose activation function:",
        ["relu", "leakyrelu", "tanh", "sigmoid"]
    )

    init_type = ask_choice(
        "Choose weight initialization:",
        ["he", "xavier", "lecun"]
    )

    loss_fn = ask_choice(
        "Choose loss function:",
        ["bce", "mse"]
    )

    optimizer = ask_choice(
        "Choose optimizer:",
        ["adam", "sgd", "rmsprop"]
    )

    layers = ask_int("Number of hidden layers", 2)
    hidden_units = ask_int("Units per hidden layer", 64)
    epochs = ask_int("Epochs", 50)
    log_every = ask_int("Log every N epochs", 5)

    dropout = ask_float("Dropout rate (0–0.5)", 0.0)
    batch_norm = ask_choice("Use BatchNorm?", ["no", "yes"]) == "yes"
    lr = ask_float("Learning rate", 0.001)

    hidden_layers = [hidden_units] * layers

    print("\n[CONFIGURATION LOCKED ]")
    print(f"Scaler={scaler}, Activation={activation}, Init={init_type}")
    print(f"Layers={hidden_layers}, Dropout={dropout}, BN={batch_norm}")
    print(f"Loss={loss_fn}, Optimizer={optimizer}, LR={lr}")
    input("\nPress ENTER to start training...")

    X_train, _, y_train, _ = load_data()

    if scaler != "none":
        X_train = get_scaler(scaler).fit_transform(X_train)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    model = Net(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        init_type=init_type
    )

    final_loss = train(
        model,
        X_train,
        y_train,
        epochs,
        lr,
        loss_fn,
        optimizer,
        log_every
    )

    print("\n=================================")
    print(f"[FINAL LOSS] {final_loss:.6f}")
    print("=================================")


if __name__ == "__main__":
    main()
