import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ResNet50 import ResNet50
from utils.data_preprocess import get_cifar10_datasets
from utils.loss_fn import CrossEntropyLoss
from scripts.train import Trainer

T = TypeVar("T")


def parse_config(config_path: str, cls: Type[T]) -> T:
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    cls_fields = {field.name for field in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return cls(**filtered_data)


@dataclass
class Args:
    data_root: str = field(
        default="./data",
        metadata={"help": "Root directory for CIFAR-10 dataset."}
    )

    download: bool = field(
        default=True,
        metadata={"help": "Whether to download CIFAR-10 dataset if not found."}
    )

    num_classes: int = field(
        default=10,
        metadata={"help": "Number of classes in CIFAR-10."}
    )

    in_channels: int = field(
        default=3,
        metadata={"help": "Number of input channels (RGB for CIFAR-10)."}
    )

    batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for training."}
    )

    lr: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for the optimizer."}
    )

    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for the optimizer."}
    )

    num_epochs: int = field(
        default=200,
        metadata={"help": "Total number of epochs for training."}
    )

    optimizer: str = field(
        default="SGD",
        metadata={"help": "Optimizer to use for training."}
    )

    momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum for SGD optimizer."}
    )

    scheduler: str = field(
        default="StepLR",
        metadata={"help": "Learning rate scheduler to use for training."}
    )

    step_size: int = field(
        default=50,
        metadata={"help": "Step size for StepLR scheduler."}
    )

    gamma: float = field(
        default=0.1,
        metadata={"help": "Gamma for StepLR scheduler."}
    )

    checkpoint_dir: str = field(
        default="./checkpoints",
        metadata={"help": "Directory to save checkpoints."}
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )

    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data loading."}
    )


def main():
    default_config_path = "config/config.json"
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parse_config(sys.argv[1], Args)
        print(f"Loaded config from {sys.argv[1]}")
    elif os.path.exists(default_config_path):
        args = parse_config(default_config_path, Args)
        print(f"Loaded default config from {default_config_path}")
    else:
        args = Args()
        print("No config file provided. Using default parameters.")

    # Get CIFAR-10 datasets
    train_dataset, val_dataset = get_cifar10_datasets(
        root=args.data_root,
        download=args.download
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model
    model = ResNet50()
    print(f"Model initialized: ResNet50 with {args.num_classes} classes")

    # Initialize loss function
    criterion = CrossEntropyLoss()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_data=train_dataset,
        val_data=val_dataset,
        **vars(args)
    )

    # Start training
    trainer.train()
    

if __name__ == "__main__":
    main()