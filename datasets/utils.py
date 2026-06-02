import os
import json
import random
import os.path as osp
from collections import defaultdict

import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch_geometric.data import Batch

from graph_builder import build_fully_connected_hetero_graph



def read_json(fpath: str) -> dict:
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj: dict, fpath: str):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def read_image(path: str) -> Image.Image:
    """Read image from path using PIL.Image, retrying on IOError."""
    if not os.path.exists(path):
        raise IOError(f'No file exists at {path}')
    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(f'Cannot read image from {path}, retrying...')

def listdir_nohidden(path: str, sort: bool = False) -> list[str]:
    """List non-hidden items in a directory."""
    items = [f for f in os.listdir(path) if not f.startswith('.')]
    if sort:
        items.sort()
    return items



class Datum:
    """
    Data instance which defines the basic attributes for an image, including
    classname and domain for compatibility with DatasetBase.
    """
    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

class DatasetBase:
    """
    A unified dataset class for managing train/val/test splits, class names,
    and other dataset-level information.
    """
    dataset_dir = ''
    domains = []

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x
        self._train_u = train_u
        self._val = val
        self._test = test

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames
    
    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')
        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)
            output.append(dataset)
        if len(output) == 1:
            return output[0]
        return output

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output













    

    

    





    





















        



def fixed_grid_crops(img: Image.Image, crop_size: int = 224, nodes=17) -> list[Image.Image]:
    """
    Generates a multi-scale grid of patches from a single image.

    The process creates:
    - 1 full-image view
    - 9 patches from a 3x3 grid
    - 4 patches from a 2x2 grid
    - 2 patches from a vertical split
    - 2 patches from a horizontal split
    
    Total patches = 1 + 9 + 4 + 2 + 2 = 18.
    """
    crops = []
    

    resize_to_crop = T.Resize(size=(crop_size, crop_size), interpolation=T.InterpolationMode.BICUBIC)



    crops.append(resize_to_crop(img))



    grid_3x3_size = crop_size * 3
    img_3x3 = T.Resize(size=(grid_3x3_size, grid_3x3_size), interpolation=T.InterpolationMode.BICUBIC)(img)
    patch_size_3x3 = grid_3x3_size // 3

    for i in range(3):
        for j in range(3):
            top = i * patch_size_3x3
            left = j * patch_size_3x3
            patch = TF.crop(img_3x3, top, left, patch_size_3x3, patch_size_3x3)
            crops.append(patch)



    grid_2x2_size = crop_size * 2
    img_2x2 = T.Resize(size=(grid_2x2_size, grid_2x2_size), interpolation=T.InterpolationMode.BICUBIC)(img)
    patch_size_2x2 = grid_2x2_size // 2

    for i in range(2):
        for j in range(2):
            top = i * patch_size_2x2
            left = j * patch_size_2x2
            patch = TF.crop(img_2x2, top, left, patch_size_2x2, patch_size_2x2)
            crops.append(patch)
            


    w, h = img_2x2.size
    

    left_half = TF.crop(img_2x2, top=0, left=0, height=h, width=w // 2)
    crops.append(resize_to_crop(left_half))
    

    right_half = TF.crop(img_2x2, top=0, left=w // 2, height=h, width=w // 2)
    crops.append(resize_to_crop(right_half))



    

    top_half = TF.crop(img_2x2, top=0, left=0, height=h // 2, width=w)
    crops.append(resize_to_crop(top_half))
    

    bottom_half = TF.crop(img_2x2, top=h // 2, left=0, height=h // 2, width=w)
    crops.append(resize_to_crop(bottom_half))


    assert len(crops) == 18, f"Expected 18 patches, but generated {len(crops)}"
    
    return crops



class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, transform=None, is_train=False):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = read_image(item.impath)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, item.label

def build_data_loader(
    data_source,
    batch_size,
    is_train=False,
    tfm=None,
    shuffle=False,
    num_workers=4,
):
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    return data_loader


class ImagePatchDataset(TorchDataset):
    def __init__(self, data_source: list[Datum], transform, patch_processor):
        self.data_source = data_source
        self.transform = transform
        self.patch_processor = patch_processor

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        image = read_image(item.impath)
        
        full_image_tensor = self.transform(image)
        
        patch_images = fixed_grid_crops(image, nodes=17)
        processed_patches = [self.patch_processor(p) for p in patch_images]
        patches_tensor = torch.stack(processed_patches)

        return {
            'image': full_image_tensor,
            'patches': patches_tensor,
            'label': torch.tensor(item.label, dtype=torch.long),
            'image_id': item.impath
        }


class GraphCollate:
    def __init__(self, visual_encoder, text_features, device):
        self.visual_encoder = visual_encoder.to(device).eval()
        self.text_features = text_features.to(device)
        self.device = device

    def __call__(self, batch: list[dict]):
        images = []
        labels = []
        graph_list = []
        
        with torch.no_grad():
            for item in batch:
                patches = item['patches'].to(self.device)
                patch_features = self.visual_encoder(patches)

                graph = build_fully_connected_hetero_graph(
                    patch_f=patch_features.cpu(),
                    text_f=self.text_features.cpu(),
                    image_id=item['image_id']
                )
                graph_list.append(graph)
                labels.append(item['label'])
                images.append(item['image'])
        
        batched_graphs = Batch.from_data_list(graph_list)
        batched_images = torch.stack(images)
        batched_labels = torch.stack(labels)
        
        return batched_graphs, batched_images, batched_labels

def build_graph_data_loader(
    data_source: list[Datum],
    batch_size: int,
    shuffle: bool,
    transform,
    visual_encoder,
    text_features,
    processor,
    device,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    dataset = ImagePatchDataset(data_source, transform=transform, patch_processor=processor)
    
    collate_fn = GraphCollate(
        visual_encoder=visual_encoder,
        text_features=text_features,
        device=device
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return data_loader
