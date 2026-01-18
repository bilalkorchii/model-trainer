import torch.nn as nn
import torch.optim as optim


LOSSES = {
    "bce": nn.BCELoss,
    "mse": nn.MSELoss
}

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop
}


def train(
    model,
    X,
    y,
    epochs,
    lr,
    loss_name,
    optimizer_name,
    log_every
):
    loss_fn = LOSSES[loss_name]()
    optimizer = OPTIMIZERS[optimizer_name](model.parameters(), lr=lr)

    print("\n[INFO] Training started\n")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(X).squeeze()
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0:
            print(f"[Epoch {epoch:03d}/{epochs}] Loss = {loss.item():.6f}")

    print("\n[INFO] Training completed ")
    return loss.item()
