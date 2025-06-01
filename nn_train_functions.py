import torch
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import time


def run_ae_epoch(model, optimizer, data_loader, loss_func,
              results, score_funcs, device, prefix=""):
    """
    Runs one epoch of training or testing.
    
    Note: This functions has the side effect of updating the results dictionary.
    
    :param model: a Pytorch model.
    :param optimizer: a Pytorch optimizer.
    :param data_loader: a Pytorch DataLoader.
    :param loss_func: a Pytorch loss function.
    :param results: a dictionary to store the results.
    :param score_funcs: a dictionary of score functions to evaluate the model.
    :param device: a string specifying the device to use.
    :param prefix: a optional string to describe the results.
    :return: a float representing the total time taken for this epoch.
    """

    # Initialize some variables
    running_loss = []
    y_true = []
    y_pred = []

    # Start the time counter
    start = time.time()

    # Loop over the batches in the data loader
    for inputs, labels in data_loader:

        # Moves the inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass.
        x_hat = model(inputs)

        # Compute loss.
        loss = loss_func(x_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save current loss
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            # Move labels & predictions back to CPU for computing predictions
            labels = labels.detach().cpu().numpy()
            x_hat = x_hat.detach().cpu().numpy()

            # Update the lists of true and predicted labels
            y_true.extend(labels.tolist())
            y_pred.extend(x_hat.tolist())

    # Stop the time counter
    end = time.time()

    # If we have a classification problem, convert to labels
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Compute the average loss and score for this epoch
    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except (ValueError, TypeError, RuntimeError):
            results[prefix + " " + name].append(float("NaN"))

    # Return the time taken for this epoch
    return end - start


def train_ae_network(model, loss_func, train_loader, val_loader=None, test_loader=None, init_lr=0.001,
                     min_lr=0.0001, epochs=50, device='cpu', score_funcs=None, lr_schedule=None, checkpoint_file=None):
    """
    Train a neural network using AdamW as a optimizer.
    
    Note: This functions has a side effect of saving the neural network training progress 
    to a checkpoint file.
    
    :param model: a Pytorch model.
    :param loss_func: a Pytorch loss function.
    :param train_loader: a DataLoader for the training set.
    :param val_loader: a DataLoader for the validation set. Defaults to None.
    :param test_loader: a DataLoader for the test set. Defaults to None.
    :param init_lr: the initial learning rate. Defaults to 0.001.
    :param min_lr: the minimum learning rate. Defaults to 0.0001.
    :param epochs: the number of epochs to train for. Defaults to 50.
    :param device: a string specifying the device to use. Defaults to 'cpu'.
    :param score_funcs: a dictionary of score functions to evaluate the model. Defaults to None.
    :param lr_schedule: a string with learning rate schedule type. Defaults to None.
    :param checkpoint_file: a string specifying the checkpoint file to save. Defaults to None.
    :return: a pandas DataFrame containing the training and evaluation results.
    """

    # Initialize the information to be tracked
    to_track = ["epoch", "total time", "train loss"]

    # If we have a val loader, we want to track the validation loss as well
    if val_loader is not None:
        to_track.append("val loss")

    # If we have a test loader, we want to track the test loss as well
    if test_loader is not None:
        to_track.append("test loss")

    # If we do not have score functions, we initialize an empty dict
    if score_funcs is None:
        score_funcs = {}

    # Track the values of the score functions
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if val_loader is not None:
            to_track.append("val " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)

    # Keep track of the total training time
    total_train_time = 0

    # Initialize a dictionary to store the results
    results = {}

    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    # Instantiate the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)

    # Instantiate the scheduler
    scheduler = None
    match lr_schedule:
        case "exp_decay":
            gamma = (min_lr / init_lr) ** (1 / epochs)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        case "step_decay":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs//4, gamma=0.3)

        case "cosine_decay":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3, eta_min=min_lr)

        case "plateau_decay":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

        case _:
            pass

    # Move the model to the device
    model.to(device)

    # Run the training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):

        # Put the model in training mode
        model = model.train()

        # Run an epoch of training
        total_train_time += run_ae_epoch(model, optimizer, train_loader, loss_func, results, score_funcs, device,
                                         prefix="train")

        # Update the results
        results["total time"].append(total_train_time)
        results["epoch"].append(epoch)

        # Run an epoch of validation
        if val_loader is not None:
            model = model.eval()
            with torch.no_grad():
                # Run an epoch of validation and save the metrics in results
                run_ae_epoch(model, optimizer, val_loader, loss_func, results, score_funcs, device, prefix="val")

        # Update the learning rate after every epoch if provided
        if scheduler is not None:
            if lr_schedule == "plateau_decay":
                if val_loader is None:
                    print("The plateau scheduler requires a validation loader to work.")
                    break
                else:
                    scheduler.step(results["val loss"][-1])
            else:
                scheduler.step()

        # Run an epoch of testing
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                # Run an epoch of testing and save the metrics in results
                run_ae_epoch(model, optimizer, test_loader, loss_func, results, score_funcs, device, prefix="test")

        # Save the results to a checkpoint file
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }, checkpoint_file)

    # Return the results as a pandas DataFrame
    return pd.DataFrame.from_dict(results)