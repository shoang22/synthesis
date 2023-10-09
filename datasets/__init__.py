import logging
from torch.utils.data import DistributedSampler, DataLoader

def get_dataset_distributed(world_size, rank, opt):
    dataset_name = opt["datasets"]["dataset_name"]
    if dataset_name == "retro":
        from .retro_dataset import RetroTrainDataset as TD
        from .retro_dataset import RetroValDataset as VD
        from .retro_dataset import RetroCollator as Collator
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")

    train_dataset = TD(opt["datasets"])
    collate_fn = Collator(opt["model"]["tokenizer"])

    if rank == 0:
        val_dataset = VD(opt["datasets"])
        logger = logging.getLogger("base")

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        shuffle=True,
        rank=rank
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt["datasets"]["train"]["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=opt["datasets"]["train"]["num_workers"],
        collate_fn=collate_fn
    )

    if rank > 0:
        return train_dataloader, None

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt["datasets"]["val"]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=opt["datasets"]["val"]["num_workers"],
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader
