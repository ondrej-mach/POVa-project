import tqdm
import torch
import optuna

import net_model
import parser
import utils

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Trains the model.
Args:
    model (torch.nn.Module): The model to train.
    trainDataLoader (torch.utils.data.DataLoader): The training data loader.
    valDataLoader (torch.utils.data.DataLoader): The validation data loader.
    opt (torch.optim.Optimizer): The optimizer to use.
    lossFn (torch.nn.Module): The loss function to use.
    iter (int): The number of iterations to train the model.
"""
def train(model, trainDataLoader, valDataLoader, opt, lossFn, iter, trainHist):

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    # loop over our epochs
    for e in range(iter):

        # set the model in training mode
        model.train()
        totalTrainLoss = 0

        # loop over the training set
        for data in tqdm.tqdm(
            trainDataLoader, desc=f"Training Epoch {e + 1}/{iter}", unit="batch"
        ):

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

        print(
            f"Epoch: {e + 1}, Train Loss: {avgTrainLoss:.4f}, Val Loss: {avgValLoss:.4f}"
        )


def calculate_split(d):
    ratios = [0.8, 0.1, 0.1]
    trCount = int(len(d) * ratios[0])
    vCount = int(len(d) * ratios[1])
    tCount = len(d) - trCount - vCount
    (trData, vData, tData) = torch.utils.data.random_split(
        d,
        [trCount, vCount, tCount],
        generator=torch.Generator().manual_seed(42),
    )
    return trData, vData, tData



def objective(trial):

    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
    epochs = 3
    folder_name = utils.create_timestamped_folder()

    # parse input data
    dataset = parser.parse_data(device)

    # calculate train/val/test sets
    trainData, valData, testData = calculate_split(dataset)

    # initialize the train, validation, and test data loaders
    trainDataLoader = torch.utils.data.DataLoader(
        trainData, shuffle=True, batch_size=batch_size
    )
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=batch_size)

    # Initialize model with suggested dropout rate
    model = net_model.CnnModel(dropout).to(device)

    # Define optimizer and loss function
    optFn = torch.optim.Adam(model.parameters(), learning_rate)
    lossFn = torch.nn.SmoothL1Loss()

    # Train the model (you would need to implement this function)
    trainHist = {"train_loss": [], "val_loss": []}
    print("Training model with learning rate:", learning_rate, "batch size:", batch_size, "dropout rate:", dropout)
    train(model, trainDataLoader, valDataLoader, optFn, lossFn, epochs, trainHist)

    utils.save_model(model, str(trial.number), trainHist, folder_name)
    
    # Return validation loss for Optuna to minimize
    return trainHist["val_loss"][-1]


def main():
    # Create a study and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    # Print best hyperparameters
    print("Best hyperparameters:", study.best_params)

if __name__ == "__main__":
    main()
