import os

import torch
import wandb
from torch.optim import lr_scheduler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.encord_dataset import EncordMaskRCNNDataset
from utils.model_libs import get_model_instance_segmentation
from utils.provider import (
    coco_remove_images_without_annotations,
    collate_fn,
    get_config,
    get_transform,
    setup_reproducibility,
    threshold_masks,
)


def train_one_epoch(model, device, data_loader, optimizer, log_freq=None):
    """Function: train_one_epoch
    Parameters:
        - model (nn.Module): The model to be trained.
        - device (str): The device to be used for training.
        - data_loader (DataLoader): The data loader containing the training data.
        - optimizer (torch.optim): The optimizer to be used for training.
        - log_freq (int): The frequency at which to log the training loss, if applicable.
    Returns:
        - None: This function does not return any values.
    Processing Logic:
        - Train the model for one epoch.
        - Move images and targets to the specified device.
        - Calculate the loss for each batch.
        - Backpropagate the loss and update the model parameters.
        - If log_freq is specified, log the training loss to WandB every log_freq batches."""
    
    model.train()

    for batch_id, (images, targets, _) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if log_freq and batch_id % log_freq == 0:
            wandb.log({"train loss": losses.cpu().item()})


@torch.inference_mode()
def evaluate(model, device, data_loader, map_metric):
    """"Evaluates the performance of a given model on a given dataset using a specified metric."
    Parameters:
        - model (nn.Module): The model to be evaluated.
        - device (torch.device): The device on which the model will be evaluated.
        - data_loader (DataLoader): The data loader containing the dataset to be evaluated on.
        - map_metric (Metric): The metric used to evaluate the model's performance.
    Returns:
        - float: The value of the specified metric for the given model on the given dataset.
    Processing Logic:
        - Set the model to evaluation mode.
        - Move images and targets to the specified device.
        - Get predictions from the model.
        - Threshold the predictions and targets.
        - Update the metric with the predictions and targets.
        - Compute the metric's value.
        - Reset the metric.
        - Return the computed metric value."""
    
    model.eval()

    for images, targets, _ in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)

        predictions = threshold_masks(predictions)
        targets = threshold_masks(targets)

        map_metric.update(preds=predictions, target=targets)

    map_metric_result = map_metric.compute()
    map_metric.reset()
    return map_metric_result


def main(params):
    """This function trains a Mask R-CNN model for object detection and segmentation on a given dataset. It takes in parameters for the dataset, model, and training settings. It returns the best mean average precision (mAP) score achieved during training.
    Parameters:
        - params (Namespace): Namespace object containing all the necessary parameters for training.
    Returns:
        - best_map (float): The best mean average precision (mAP) score achieved during training.
    Processing Logic:
        - Sets up reproducibility with a given seed.
        - Initializes variables for tracking the best mAP score, the last epoch, and the early stopping counter.
        - Checks for the availability of a CUDA device and sets the device accordingly.
        - Creates a dataset object for the training data and gets the number of classes in the dataset.
        - Filters out images without annotations from the training dataset.
        - Creates a dataset object for the validation data and filters out images without annotations.
        - Defines training and validation data loaders.
        - Gets the Mask R-CNN model and moves it to the appropriate device.
        - Constructs an optimizer and a learning rate scheduler if specified.
        - Initializes mean average precision (mAP) metrics for training and validation.
        - Trains the model for the specified number of epochs.
        - Evaluates the model's performance on the training and validation data.
        - Logs the performance metrics to WandB if enabled.
        - Saves the best model based on the validation mAP score.
        - Stops training early if the validation mAP score does not improve for a specified number of epochs.
        - Saves the final model and prints a message indicating that training has finished."""
    
    setup_reproducibility(35)

    best_map = 0
    last_epoch = 0
    early_stop_counter = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = EncordMaskRCNNDataset(
        img_folder=params.data.train_data_folder,
        ann_file=params.data.train_ann,
        transforms=get_transform(train=True),
    )
    num_classes = len(dataset.coco.cats) + 1  # due to background

    print(f"Total training images before filtering: {len(dataset)}")
    dataset = coco_remove_images_without_annotations(dataset)
    print(f"Total training images after filtering: {len(dataset)}")

    dataset_validation = EncordMaskRCNNDataset(
        img_folder=params.data.validation_data_folder,
        ann_file=params.data.validation_ann,
        transforms=get_transform(train=False),
    )
    print(f"Total validation images before filtering: {len(dataset_validation)}")
    dataset_validation = coco_remove_images_without_annotations(dataset_validation)
    print(f"Total validation images after filtering: {len(dataset_validation)}")

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.train.batch_size,
        shuffle=True,
        num_workers=params.train.num_worker,
        collate_fn=collate_fn,
    )

    data_loader_validation = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=1,
        shuffle=False,
        num_workers=params.train.num_worker,
        collate_fn=collate_fn,
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, fine_tuning=True)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_params, lr=params.train.learning_rate)

    if params.train.use_lr_scheduler:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.2,
            patience=params.train.lr_scheduler_patience,
            threshold=0.0001,
            verbose=True,
        )

    train_map_metric = MeanAveragePrecision(iou_type="segm").to(device)
    val_map_metric = MeanAveragePrecision(iou_type="segm").to(device)

    for epoch in range(params.train.max_epoch):
        last_epoch = epoch
        print(f"Epoch: {epoch}")
        train_one_epoch(model, device, data_loader, optimizer, log_freq=10)

        if epoch % params.logging.performance_tracking_interval == 0:
            if params.logging.log_train_map:
                train_map = evaluate(model, device, data_loader, train_map_metric)
            val_map = evaluate(model, device, data_loader_validation, val_map_metric)

            if params.train.use_lr_scheduler:
                scheduler.step(val_map["map"])

            if params.logging.wandb_enabled:
                train_map_logs = {}
                if params.logging.log_train_map:
                    train_map_logs = {
                        f"train/{k}": v.item() for k, v in train_map.items()
                    }
                val_map_logs = {f"val/{k}": v.item() for k, v in val_map.items()}
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                        **train_map_logs,
                        **val_map_logs,
                    }
                )

            val_map_average = val_map["map"].cpu().item()

            if val_map_average > best_map * (1 + 0.0001):
                early_stop_counter = 0
                best_map = val_map_average
                print("overwriting the best model!")

                if params.logging.wandb_enabled:
                    wandb.run.summary["best map"] = best_map
                    torch.save(
                        model.state_dict(),
                        os.path.join(wandb.run.dir, "best_maskrcnn.ckpt"),
                    )
                else:
                    torch.save(model.state_dict(), "weights/best_maskrcnn.ckpt")
            else:
                early_stop_counter += 1

            if early_stop_counter >= params.train.early_stopping_thresh:
                print("Early stopping at: " + str(epoch))
                break

    if params.logging.wandb_enabled:
        torch.save(
            model.state_dict(),
            os.path.join(wandb.run.dir, f"epoch_{last_epoch}_maskrcnn.ckpt"),
        )
    else:
        torch.save(model.state_dict(), f"weights/epoch_{last_epoch}_maskrcnn.ckpt")

    print("Training finished")


if __name__ == "__main__":
    params = get_config("config.ini")
    if params.logging.wandb_enabled:
        wandb.init(project=params.logging.wandb_project, save_code=True)
        wandb.run.name = (
            os.path.basename(__file__)[:-3] + "_" + wandb.run.name.split("-")[2]
        )
        wandb.run.save()

        config = wandb.config
        config.train_data_folder = params.data.train_data_folder
        config.train_ann_file = params.data.train_ann
        config.validation_data_folder = params.data.validation_data_folder
        config.validation_ann_fie = params.data.validation_ann
        config.lr = params.train.learning_rate
        config.bs = params.train.batch_size
        config.num_worker = params.train.num_worker

    main(params)

    if params.logging.wandb_enabled:
        wandb.run.finish()
