# from torchvision import transforms
from .transforms import transforms
from torch.utils.data import DataLoader
from .mydataset import dataset as my_dataset, dataset_with_mask
import torchvision
import torch
import numpy as np

def data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    # tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
    #                                  transforms.CenterCrop(crop_size), #321
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize(mean_vals, std_vals)
    #                                  ])

    tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 356
                                     transforms.RandomCrop(crop_size),  # 321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])


    tsfm_test = transforms.Compose([transforms.Resize(input_size),  # 256
                                     transforms.CenterCrop(crop_size),  # 224
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    img_train = my_dataset(args.train_list, root_dir=args.img_dir,
                           transform=tsfm_train, with_path=True)

    img_test = my_dataset(args.test_list, root_dir=args.img_dir,
                          transform=tsfm_test, with_path=test_path)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def data_loader2(args, test_path=False):
    mean_vals = [103.939, 116.779, 123.68]
    mean_vals = torch.FloatTensor(mean_vals).unsqueeze(dim=1).unsqueeze(dim=1)

    tsfm_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: x.sub_(mean_vals)
        )
    ])


    tsfm_test = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Lambda(
                                        lambda x: x.sub_(mean_vals)
                                    )
                                    ])

    img_train = my_dataset(args.train_list, root_dir=args.img_dir,
                           transform=tsfm_train, with_path=True)

    # img_test = my_dataset(args.train_list, root_dir=args.img_dir,
    #                       transform=tsfm_train, with_path=test_path)

    img_test = my_dataset(args.test_list, root_dir=args.img_dir,
                          transform=tsfm_test, with_path=test_path)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader
