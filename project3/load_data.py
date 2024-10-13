def load_data(batch_size=64, num_workers=4, validation_split=0.1, shuffle_option=True):
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
    import numpy as np

    # 数据预处理和转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 你可以在这里添加更多的图像增强或预处理操作
    ])

    # 加载 CIFAR10 训练和测试数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    # 创建训练集的索引，并进行训练和验证集的划分
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    # 打乱索引以随机分配训练和验证集
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    # 根据 shuffle_option 参数决定是否打乱训练集
    if shuffle_option:
        train_sampler = SubsetRandomSampler(train_idx)  # 随机采样
    else:
        train_sampler = SequentialSampler(train_idx)  # 顺序采样

    # 验证集始终使用顺序采样
    val_sampler = SequentialSampler(val_idx)

    # 创建 DataLoader 用于训练、验证和测试集
    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             sampler=train_sampler, num_workers=num_workers)

    valloader = DataLoader(train_dataset, batch_size=batch_size,
                           sampler=val_sampler, num_workers=num_workers)

    testloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return trainloader, testloader, valloader