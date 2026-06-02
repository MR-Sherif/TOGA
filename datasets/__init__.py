from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .dogs import Dogs
from .birds import Birds
from .imagenet import ImageNet


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "aircrafts": FGVCAircraft,
                "food101": Food101,
                "flowers": OxfordFlowers,
                "cars": StanfordCars,
                "birds": Birds,
                "dogs": Dogs,
                "imagenet": ImageNet
                }


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)
