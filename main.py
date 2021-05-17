import subprocess

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar10",
#                  "--model","vgg16",
#                  "--optimizer", "sgd",
#                  "--lr", "0.1",
#                  "--prefix", "Dec18_08-22-32",
#                  "--resume","pretrain/vgg16_cifar10.pth"
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_vgg16_cifar10_Dec18_08-22-32",
#                  "--lam","0.1",
#                  "--lr","0.1",
#                  "--sharescale"
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar10",
#                  "--model","preactresnet18",
#                  "--optimizer", "sgd",
#                  "--lr", "0.1",
#                  "--resume","pretrain/vgg16_preactresnet18.pth"
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_preactresnet18_cifar10_Dec20_20-58-11",
#                  "--lam","0.1",
#                  "--lr","0.1",
#                  "--sharescale"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar10_Dec18_08-22-32",
#                  "--suffix", "_[0.100_91.340_1.801]",
#                  "--simulation", "curve"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet18_cifar10_Dec20_20-58-11",
#                  "--batch_size","20",
#                  "--suffix", "_[0.100_93.060_2.068]",
#                  "--simulation", "curve"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet18_cifar10_Dec20_20-58-11",
#                  "--suffix", "_[0.100_93.060_2.068]",
#                  "--simulation", "acc"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "normal_train_vgg16_cifar10_Dec22_15-30-22",
#                  "--batch_size","20",
#                  "--suffix", "x0.800000",
#                  "--simulation", "curve",
#                  "--loadmodel"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "normal_train_vgg16_cifar10_Dec22_15-30-22",
#                  "--batch_size","50",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--loadmodel"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "normal_train_vgg16_cifar10_Dec22_15-30-22",
#                  "--batch_size","50",
#                  "--suffix", "",
#                  "--simulation", "power",
#                  "--loadmodel"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar10_Dec18_08-22-32",
#                  "--batch_size","50",
#                  "--suffix", "_[0.100_91.340_1.801]",
#                  "--simulation", "power"
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar100",
#                  "--model","resnet50",
#                  "--optimizer", "sgd",
#                  "--lr", "0.0001",
#                  "--resume","pretrain/resnet50_cifar100.pth"
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar100",
#                  "--model","resnet101",
#                  "--optimizer", "sgd",
#                  "--lr", "0.0001",
#                  "--resume","pretrain/resnet101_cifar100.pth"
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar100",
#                  "--model","preactresnet18",
#                  "--optimizer", "sgd",
#                  "--lr", "0.1",
#                  "--resume","pretrain/preactresnet18_cifar100.pth"
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_preactresnet18_cifar100_Jan03_13-25-15",
#                  "--lam","1",
#                  "--lr","0.01",
#                  "--sharescale",
#                  "--acc_tolerance","0.1"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet18_cifar100_Jan03_13-25-15",
#                  "--batch_size","1000",
#                  "--suffix", "_[1.000_69.430_3.777]",
#                  "--T","1000",
#                  "--simulation", "acc"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet18_cifar100_Jan03_13-25-15",
#                  "--batch_size","1000",
#                  "--suffix", "",
#                  "--T","1000",
#                  "--simulation", "acc"
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar100",
#                  "--model","vgg16",
#                  "--optimizer", "sgd",
#                  "--lr", "0.1",
#                  "--resume","pretrain/vgg16_cifar100.pth"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar100_Jan04_00-26-06",
#                  "--batch_size","1000",
#                  "--suffix", "",
#                  "--T","1000",
#                  "--simulation", "acc"
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_vgg16_cifar100_Jan04_00-26-06",
#                  "--lam","0.1",
#                  "--lr","0.1",
#                  "--sharescale",
#                  "--acc_tolerance","0.1",
#                  "--init",'2.5'
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar100_Jan04_00-26-06",
#                  "--batch_size","1000",
#                  "--suffix", "_[0.100_66.960_3.958]",
#                  "--T","1000",
#                  "--simulation", "acc"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar100_Jan04_00-26-06",
#                  "--batch_size","1000",
#                  "--suffix", "_[0.100_59.610_1.883]",
#                  "--T","1000",
#                  "--simulation", "acc"
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "mnist",
#                  "--model","cnn",
#                  "--optimizer", "sgd",
#                  "--lr", "0.01",
#                  "--batch_size","128"
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_cnn_mnist_Jan04_11-20-59",
#                  "--lam","1",
#                  "--lr","0.01",
#                  "--sharescale",
#                  "--acc_tolerance","0.1",
#                  "--batch_size","128",
#                  "--init",'3'
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "mnist",
#                  "--model","alexnet",
#                  "--optimizer", "sgd",
#                  "--lr", "0.01",
#                  "--batch_size","128"
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_alexnet_mnist_Jan04_13-04-40",
#                  "--lam","0.1",
#                  "--lr","0.01",
#                  "--sharescale",
#                  "--acc_tolerance","0.1",
#                  "--batch_size","128",
#                  "--init",'3'
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_alexnet_mnist_Jan04_13-04-40",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--T","500",
#                  "--simulation", "acc"
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_alexnet_mnist_Jan04_13-04-40",
#                  "--batch_size","100",
#                  "--suffix", "_[0.100_98.800_1.822]",
#                  "--T","500",
#                  "--simulation", "acc"
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_cnn_mnist_Jan04_11-20-59",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--T","500",
#                  "--simulation", "acc"
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_cnn_mnist_Jan04_11-20-59",
#                  "--batch_size","100",
#                  "--suffix", "_[1.000_96.230_2.059]",
#                  "--T","500",
#                  "--simulation", "acc"
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_cnn_mnist_Jan04_11-20-59",
#                  "--batch_size","100",
#                  "--suffix", "_[1.000_94.000_1.696]",
#                  "--T","500",
#                  "--simulation", "acc"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "normal_train_vgg16_cifar10_Dec24_09-43-27",
#                  "--batch_size","100",
#                  "--suffix", "_robust",
#                  "--simulation", "acc",
#                  "--T","500",
#                  "--loadmodel"
#                  ])


# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "normal_train_preactresnet18_cifar10_Jan05_09-06-38",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","500",
#                  "--loadmodel"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "normal_train_preactresnet18_cifar10_Jan05_09-06-38",
#                  "--batch_size","100",
#                  "--suffix", "_robust",
#                  "--simulation", "acc",
#                  "--T","500",
#                  "--loadmodel"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar10_Dec18_08-22-32",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","500",
#                  ])


# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar10_Dec18_08-22-32",
#                  "--batch_size","100",
#                  "--suffix", "_[0.100_92.810_3.813]",
#                  "--simulation", "acc",
#                  "--T","500",
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar10_Dec18_08-22-32",
#                  "--suffix", "_[0.100_92.810_3.813]",
#                  "--simulation", "curve"
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar10_Dec18_08-22-32",
#                  "--batch_size","50",
#                  "--suffix", "_[0.100_92.810_3.813]",
#                  "--simulation", "power"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_vgg16_cifar100_Jan04_00-26-06",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","1500",
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet18_cifar100_Jan03_13-25-15",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","1000",
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet18_cifar10_Dec20_20-58-11",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","500",
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_cnn_mnist_Jan04_11-20-59",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","500",
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_alexnet_mnist_Jan04_13-04-40",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","500",
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet50_cifar100_Dec28_13-15-42",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","1500",
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar100",
#                  "--model","preactresnet34",
#                  "--optimizer", "sgd",
#                  "--lr", "0.1",
#                  "--resume","pretrain/preactresnet34_cifar100.pth"
#                  ])

# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet34_cifar100_Jan06_19-14-12",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","4000",
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_preactresnet34_cifar100_Jan06_19-14-12",
#                  "--lam","1",
#                  "--lr","0.01",
#                  "--sharescale",
#                  "--acc_tolerance","0.1",
#                  "--batch_size","128",
#                  "--init",'4'
#                  ])

# subprocess.call(["python", "para_train.py",
#                  "--dataset", "cifar100",
#                  "--model","preactresnet50",
#                  "--optimizer", "sgd",
#                  "--lr", "0.1",
#                  "--resume","pretrain/preactresnet50_cifar100.pth"
#                  ])

# subprocess.call(["python", "fast_train.py",
#                  "--pretrain", "train_preactresnet50_cifar100_Jan07_12-44-19",
#                  "--lam","0.5",
#                  "--lr","0.0025",
#                  "--sharescale",
#                  "--acc_tolerance","0.1",
#                  "--batch_size","32",
#                  "--init",'4'
#                  ])




# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet34_cifar100_Jan06_19-14-12",
#                  "--batch_size","500",
#                  "--suffix", "_[1.000_62.240_2.838]",
#                  "--simulation", "acc",
#                  "--T","4000",
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet50_cifar100_Jan07_12-44-19",
#                  "--batch_size","100",
#                  "--suffix", "",
#                  "--simulation", "acc",
#                  "--T","4000",
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet50_cifar100_Jan07_12-44-19",
#                  "--batch_size","100",
#                  "--suffix", "_[0.500_65.450_3.262]",
#                  "--simulation", "acc",
#                  "--T","4000",
#                  ])
#
# subprocess.call(["python", "evaluate.py",
#                  "--pretrain", "train_preactresnet50_cifar100_Jan07_12-44-19",
#                  "--batch_size","100",
#                  "--suffix", "_[0.500_63.460_2.856]",
#                  "--simulation", "acc",
#                  "--T","4000",
#                  ])