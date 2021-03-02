from model import ClassificationModel
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from config import batch_size, image_size, max_epochs, train_dataset_path, valid_dataset_path, val_pct


def get_data_loaders_from_dataset(train_ds, valid_ds):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_dl, valid_dl


def main():
    # Image Augmentations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])

    valid_data_loader = None
    train_data_loader = None

    if valid_dataset_path:
        val_dataset = ImageFolder(valid_dataset_path, transform=transform)
        train_dataset = ImageFolder(train_dataset_path, transform=transform)
        train_data_loader, valid_data_loader = get_data_loaders_from_dataset(train_dataset, val_dataset)
    else:
        image_ds = ImageFolder(train_dataset_path, transform=transform)
        print(image_ds.classes)
        val_size = int(val_pct * len(image_ds))
        train_size = len(image_ds) - val_size
        train_dataset, val_dataset = random_split(image_ds, [train_size, val_size])
        train_data_loader, valid_data_loader = get_data_loaders_from_dataset(train_dataset, val_dataset)

    flowers_model = ClassificationModel()
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs)
    trainer.fit(flowers_model, train_dataloader=train_data_loader, val_dataloaders=valid_data_loader)


if __name__ == '__main__':
    main()