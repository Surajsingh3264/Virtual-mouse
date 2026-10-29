import copy
import torch
from torch import nn
import matplotlib.pyplot as plt

# 1. Make the experiment reproducible
torch.manual_seed(42)

# 2. Generate a synthetic dataset
n_samples = 1000

X = torch.linspace(-3 * torch.pi, 3 * torch.pi, n_samples)
X = X.reshape(-1, 1)

noise = 0.15 * torch.randn_like(X)
y = torch.sin(X) + noise

# 3. Shuffle and split the dataset
indices = torch.randperm(n_samples)

train_end = int(0.70 * n_samples)
val_end = int(0.85 * n_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# 4. Build a neural network
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 5. Set up training
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003,
    weight_decay=0.0001
)

# 6. Configure early stopping
max_epochs = 500
patience = 40

best_val_loss = float("inf")
best_epoch = 0
best_weights = None
epochs_without_improvement = 0

train_history = []
val_history = []

# 7. Train the model
for epoch in range(max_epochs):

    # Training phase
    model.train()

    train_predictions = model(X_train)
    train_loss = loss_function(train_predictions, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_function(val_predictions, y_val)

    train_history.append(train_loss.item())
    val_history.append(val_loss.item())

    # Save the best model so far
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_epoch = epoch + 1
        best_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Stop if validation loss has not improved
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# 8. Restore the best validation checkpoint
model.load_state_dict(best_weights)

# 9. Evaluate on the test set
model.eval()

with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = loss_function(test_predictions, y_test)

print(f"Best epoch: {best_epoch}")
print(f"Best validation MSE: {best_val_loss:.4f}")
print(f"Test MSE: {test_loss.item():.4f}")

# 10. Plot training and validation loss
plt.figure(figsize=(9, 5))
plt.plot(train_history, label="Training loss")
plt.plot(val_history, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

# 11. Plot predictions against the underlying function
x_plot = torch.linspace(-3 * torch.pi, 3 * torch.pi, 500)
x_plot = x_plot.reshape(-1, 1)

with torch.no_grad():
    predictions = model(x_plot)

plt.figure(figsize=(9, 5))
plt.scatter(X.numpy(), y.numpy(), s=8, alpha=0.25, label="Noisy data")
plt.plot(x_plot.numpy(), torch.sin(x_plot).numpy(),
         label="Underlying sine function")
plt.plot(x_plot.numpy(), predictions.numpy(),
         label="Neural network prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Learning a Nonlinear Relationship")
plt.legend()
plt.tight_layout()
plt.show()
