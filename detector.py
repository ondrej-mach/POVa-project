import tqdm
import torch
import optuna

import net_model
import parser
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, trainDataLoader, valDataLoader, opt, lossFn, iter, trainHist):

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    stop_val_loss = 0.005

    # loop over our epochs
    for epoch in range(iter):

        # set the model in training mode
        model.train()
        totalTrainLoss = 0

        # loop over the training set
        for data in tqdm.tqdm(trainDataLoader, desc=f"Training Epoch {epoch + 1}/{iter}", unit="batch"):

            # Load image and bounding box from data structure
            img = data["img"]
            bbox = data["box"]

            pred = model(img)

            loss = lossFn(pred, bbox)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss.item()

        scheduler.step()

        model.eval()
        totalValLoss = 0

        # switch off autograd for evaluation
        with torch.no_grad():
            for data in tqdm.tqdm(valDataLoader, desc="Validating", unit="batch"):
                img_val = data["img"]
                bbox_val_tensor = data["box"]

                pred_val = model(img_val)
                val_loss = lossFn(pred_val, bbox_val_tensor)

                totalValLoss += val_loss.item()

        avgTrainLoss = totalTrainLoss / len(trainDataLoader)
        avgValLoss = totalValLoss / len(valDataLoader)

        trainHist["train_loss"].append(avgTrainLoss)
        trainHist["val_loss"].append(avgValLoss)

        print(f"Epoch: {epoch + 1}, Train Loss: {avgTrainLoss:.4f}, Val Loss: {avgValLoss:.4f}")

        # Early stopping to avoid overfitting
        if avgValLoss < stop_val_loss:
            print("Validation loss below threshold. Stopping training to avoid overfitting.")
            break


def evaluate_model(model, testDataLoader, lossFn):
    # Set the model to evaluation mode
    model.eval()
    totalTestLoss = 0
    num_samples = 0

    with torch.no_grad():
        # Loop over the training set
        for data in tqdm.tqdm(testDataLoader, desc=f"Testing", unit="batch"):

            image = data["img"]
            bbox = data["box"]

            # Forward pass: make predictions using the model
            pred = model(image)

            # Calculate loss (assuming bbox prediction)
            test_loss = lossFn(pred, bbox)
            totalTestLoss += test_loss.item()
            num_samples += 1

    avgTestLoss = totalTestLoss / num_samples if num_samples > 0 else float("inf")
    print(f"Average Test Loss: {avgTestLoss:.4f}")


def objective(trial):

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 32)
    dropout = trial.suggest_float("dropout_rate", 0.1, 0.3)
    epochs = 20
    folder_name = utils.create_timestamped_folder()

    # Parse input data
    dataset = parser.parse_data(device)

    # Calculate train/val/test sets
    trainData, valData, testData = utils.calculate_split(dataset)

    # Initialize the train, validation, and test data loaders
    trainDataLoader = torch.utils.data.DataLoader(
        trainData, shuffle=True, batch_size=batch_size
    )
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=batch_size)
    testDataLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size)

    # Initialize model with suggested dropout rate
    print("Moving model to device:", device)
    model = net_model.CnnModel(dropout).to(device)

    # Define optimizer and loss function
    optFn = torch.optim.Adam(model.parameters(), learning_rate)
    lossFn = torch.nn.SmoothL1Loss()

    trainHist = {"train_loss": [], "val_loss": []}

    # Train the model
    print("Training model with learning rate:", learning_rate, "batch size:", batch_size, "dropout rate:", dropout)
    train(model, trainDataLoader, valDataLoader, optFn, lossFn, epochs, trainHist)

    # Evaluate the model using the test data
    evaluate_model(model, testDataLoader, lossFn)

    # Save the model and training history
    utils.save_model(model, trainHist, folder_name)

    # Return validation loss for Optuna to minimize
    return trainHist["val_loss"][-1]


def main():
    if torch.cuda.is_available():
        print("Using GPU")

    # Create a study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)

    # Print best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)


if __name__ == "__main__":
    main()
