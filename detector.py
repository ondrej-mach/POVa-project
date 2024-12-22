import tqdm
import torch
import argparse
import optuna
from optuna.trial import TrialState

import net_model
import parser
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, trainDataLoader, valDataLoader, opt, lossFn, iter, trainHist):

    # loop over our epochs
    for epoch in range(iter):
        totalTrainLoss = 0

        # training mode
        model.train()

        # loop over the training set
        for data in tqdm.tqdm(trainDataLoader, desc=f"Training Epoch {epoch + 1}/{iter}", unit="batch"):

            # load image and bounding box
            img = data["img"]
            bbox = data["box"]

            # forward pass
            pred = model(img)

            # calculate loss
            loss = lossFn(pred, bbox)

            # zero the gradients
            opt.zero_grad()

            # backward pass
            loss.backward()

            # update the weights
            opt.step()

            totalTrainLoss += loss.item()

        # evaluation mode
        model.eval()
        with torch.no_grad():
            totalValLoss = 0

            # loop over the validation set
            for data in tqdm.tqdm(valDataLoader, desc="Validating", unit="batch"):

                img_val = data["img"]
                bbox_val_tensor = data["box"]

                # forward pass
                pred_val = model(img_val)

                # calculate loss
                val_loss = lossFn(pred_val, bbox_val_tensor)
                totalValLoss += val_loss.item()

        avgTrainLoss = totalTrainLoss / len(trainDataLoader)
        avgValLoss = totalValLoss / len(valDataLoader)

        trainHist["train_loss"].append(avgTrainLoss)
        trainHist["val_loss"].append(avgValLoss)

        print(f"Epoch: {epoch + 1}, Train Loss: {avgTrainLoss:.4f}, Val Loss: {avgValLoss:.4f}")

    return avgValLoss

def objective(trial):
 
    # Parse the number of epochs
    p = argparse.ArgumentParser()
    p.add_argument("epochs", type=int, help="Number of epochs")
    args = p.parse_args()
    max_epochs = args.epochs

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 24)
    dropout = trial.suggest_float("dropout_rate", 0.0001, 0.2)
    epochs = trial.suggest_int("epochs", 10, max_epochs)

    # Parse input data
    dataset = parser.parse_data(device)
    
    # Calculate train/val/test sets
    trainData, valData, testData = utils.calculate_split(dataset)

    # Initialize the train, validation, and test data loaders
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True)
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=batch_size)
    testDataLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size)

    # Initialize model with suggested dropout rate
    model = net_model.CnnModel(dropout).to(device)

    # Define optimizer and loss function
    optFn = torch.optim.Adam(model.parameters(), learning_rate)
    lossFn = torch.nn.SmoothL1Loss()

    trainHist = {"train_loss": [], "val_loss": []}

    # Train the model
    print("Training model with learning rate:", learning_rate, "batch size:", batch_size, "dropout rate:", dropout)
    avgValLoss = train(model, trainDataLoader, valDataLoader, optFn, lossFn, epochs, trainHist)

    if avgValLoss < 0.01:
        # Save the model and training history
        folder_name = utils.create_timestamped_folder()
        utils.save_model(model, trainHist, folder_name, False)

    return trainHist["val_loss"][-1]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=150)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
