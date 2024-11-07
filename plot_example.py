import torchvision

from loading_data.load_data import LoadData
from result.plot import imshow

poison_types = ['sig']
data_names = ["mnist", "cifar10"]
path = "./examples/"

for data_name in data_names:
    for poison_type in poison_types:
        data = LoadData(data_name, poison_type)
        _, test_loader, poison_test_loader = data.get_date(batch_size=1)
        dataiter = iter(poison_test_loader)
        images, _ = next(dataiter)

        imshow(torchvision.utils.make_grid(images, nrow=2), path+f"{data_name}-{poison_type}")
