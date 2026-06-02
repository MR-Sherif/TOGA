import os, pickle
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
from itertools import chain

import json
import cv2
import torch
from torch_geometric.data import Data, Batch
from torchvision.tv_tensors import BoundingBoxes
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
_EDGE_CACHE = {}

def get_edge(num_nodes, samples):
    """Fully connected graph edges, cached by node count."""
    if num_nodes not in _EDGE_CACHE:
        edges = torch.combinations(torch.arange(num_nodes), 2)
        edges = torch.cat([edges, edges.flip(1)], dim=0).t()
        _EDGE_CACHE[num_nodes] = edges
    return _EDGE_CACHE[num_nodes].repeat(1, samples)

save_plot = True
def plot(images, plot):

    n_rows = 6
    n_cols = 6


    plt.figure(figsize=(7 * n_cols, 7))

    for i, img in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        if(i == 15):
            plt.title(f"Image center_crop")
        else:
            plt.title(f"Image {i+1}")

    plt.tight_layout()
    plt.savefig(f"{plot}.jpg")

    
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
    
def load_pickle_data(file):
        with open(file, "rb") as f:
            try:
                data = pickle.load(f)
            except:
                data = CPU_Unpickler(f).load()
        return data
    


def fixed_grid_crops(img, crop_size=224, nodes=26):
    """
    Generate 17 deterministic crops from a 336×336 image:
    - 4 corner crops (224×224)
    - 1 center crop (224×224)
    - 4 overlapping shifted crops (224×224)
    - 4 quadrant crops (168×168 resized to 224×224)
    - 2 horizontal halves (336×168 resized to 224×224)
    - 2 vertical halves (168×336 resized to 224×224) ← optional if needed later
    """
    resize = T.Resize(size=(336, 336),  interpolation=T.InterpolationMode.BICUBIC)
    resize224 = T.Resize(size=(224, 224),  interpolation=T.InterpolationMode.BICUBIC)
    img = resize(img)
    w, h = img.size
    assert w == 336 and h == 336, f"Expected image of size 336x336, but got {w}x{h}"
    stride = crop_size // 2
    crops = []
    

    crops.append(resize224(img))

    corners = [
        (0, 0), (w - crop_size, 0),
        (0, h - crop_size), (w - crop_size, h - crop_size)
    ]
    for x, y in corners:
        crops.append(TF.crop(img, top=y, left=x, height=crop_size, width=crop_size))

    crops.append(resize224(img))
    

    shifts = [
        (stride, 0), (0, stride),
        (2 * stride, stride), (stride, 2 * stride)
    ]
    for x, y in shifts:
        i = TF.crop(img, top=y, left=x, height=stride, width=stride)
        i = resize(i)
        crops.append(i)



    quadrant_size = 168
    resize_transform = T.Resize((224, 224))
    quadrants = [
        (0, 0), (w - quadrant_size, 0),
        (0, h - quadrant_size), (w - quadrant_size, h - quadrant_size)
    ]
    for x, y in quadrants:
        q_crop = TF.crop(img, top=y, left=x, height=168, width=168)
        crops.append(resize_transform(q_crop))


    horizontal_halves = [(0, 0), (0, h // 2)]
    for x, y in horizontal_halves:
        h_half_crop = TF.crop(img, top=y, left=x, height=168, width=336)
        crops.append(resize_transform(h_half_crop))


    vertical_halves = [(0, 0), (w // 2, 0)]
    for x, y in vertical_halves:
        v_half_crop = TF.crop(img, top=y, left=x, height=336, width=168)

        crops.append(resize_transform(v_half_crop))
    

    if(nodes == 26):
        tile_size = 112
        for i in range(3):
            for j in range(3):
                x = j * tile_size
                y = i * tile_size
                tile = TF.crop(img, top=y, left=x, height=tile_size, width=tile_size)
                crops.append(resize_transform(tile))
                
    if(nodes == 32):
        tile_size = 112
        for i in range(3):
            for j in range(3):
                x = j * tile_size
                y = i * tile_size
                tile = TF.crop(img, top=y, left=x, height=tile_size, width=tile_size)
                crops.append(resize_transform(tile))

        tile_size = 84
        start_x = 0
        start_y = 0
        for i in range(3):
            start_x = 0
            for j in range(2):
                width = tile_size*3
                height = tile_size*2
                x = start_x
                y = start_y
                tile = TF.crop(img, top=y, left=x, height=height, width=width)
                crops.append(resize_transform(tile))
                start_x += tile_size    
            start_y += tile_size
            
    if(nodes==38):
        tile_size = 112
        for i in range(3):
            for j in range(3):
                x = j * tile_size
                y = i * tile_size
                tile = TF.crop(img, top=y, left=x, height=tile_size, width=tile_size)
                crops.append(resize_transform(tile))

        tile_size = 84
        start_x = 0
        start_y = 0
        for i in range(3):
            start_x = 0
            for j in range(2):
                width = tile_size*3
                height = tile_size*2
                x = start_x
                y = start_y
                tile = TF.crop(img, top=y, left=x, height=height, width=width)
                crops.append(resize_transform(tile))
                start_x += tile_size    
            start_y += tile_size
            

        tile_size = 84
        start_x = 0
        start_y = 0
        for i in range(2):
            start_x = 0
            for j in range(3):
                width = tile_size*2
                height = tile_size*3
                x = start_x
                y = start_y
                tile = TF.crop(img, top=y, left=x, height=height, width=width)
                crops.append(resize_transform(tile))
                start_x += tile_size    
            start_y += tile_size

    if(nodes==26):
        assert len(crops) == 26+1, f"Expected 26 crops but got {len(crops)}"
    elif(nodes==32):
        assert len(crops) == 32+1, f"Expected 32 crops but got {len(crops)}"
    elif(nodes==38):
        assert len(crops) == 38+1, f"Expected 38 crops but got {len(crops)}"
    else:
        assert len(crops) == 17+1, f"Expected 17 crops but got {len(crops)}"

    
    return crops

def fixed_crops_8(img, clip_processor, crop_size=224, nodes=8):
    """Generate structured crops from image (original + quadrants)."""
    resize = T.Resize((336, 336), interpolation=T.InterpolationMode.BICUBIC)
    resize224 = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)

    img = resize(img)
    w, h = img.size
    assert (w, h) == (336, 336)

    crops = [resize224(img)]


    crops.extend([
        TF.crop(img, 0, 0, 336, 168),
        TF.crop(img, 0, 168, 336, 168)
    ])


    crops.extend([
        TF.crop(img, 0, 0, 168, 336),
        TF.crop(img, 168, 0, 168, 336)
    ])


    for x, y in [(0, 0), (168, 0), (0, 168), (168, 168)]:
        crops.append(resize224(TF.crop(img, y, x, 168, 168)))

    if nodes == 8:
        assert len(crops) == 9, f"Expected 9 crops, got {len(crops)}"
    crops = [ clip_processor(c) for c in crops]
    return crops

def random_crops(img):
    resize = T.Resize(size=(336, 336),  interpolation=T.InterpolationMode.BICUBIC)
    img = resize(img)
    crops = []
    for i in range(10):
        x = random.randrange(1, 113)
        y = random.randrange(1,113)
        crops.append(TF.crop(img, top=y, left=x, height=224, width=224))
    return crops


    
def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
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
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
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
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )



    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
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
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output

class RandomRotateWithBoxes:
    def __init__(self, degrees, expand=True, p=0.5):
        self.degrees = degrees
        self.expand = expand
        self.p = p

    def __call__(self, image, target):

        if torch.rand(1) < self.p:
            angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
            image = F.rotate(image, angle, expand=self.expand)
            target["boxes"] = F.rotate_bounding_boxes(
                target["boxes"], angle, 
                format="XYXY", 
                image_size=(image.shape[1], image.shape[2]), 
                expand=self.expand
            )
        return image, target


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1, processor=None):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train

        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0
        self.processor = processor
        self.resize336 = A.Compose([
        A.Resize(width=336, height=336, interpolation= 4, p=1.0),
        ])
        self.transform_box = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.SafeRotate(limit=25, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1),
            A.GaussianBlur(p=0.2),


            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
        ])

        
        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )


        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)
    
    def __len__(self):
        return len(self.data_source)
    
    def check_bb(self, img_path):
        bb_loc = img_path.replace("data", "bb_masks").replace("jpg", "pkl")
        if(os.path.exists(bb_loc)):
            return bb_loc
        else:
            return False
        

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }
        
        
        img0 = read_image(item.impath)
        w, h = img0.size
        if(self.is_train and self.processor != None):
            img_np = np.array(img0)
            augmented = self.transform_box(image=img_np)
            transformed_img = Image.fromarray(augmented["image"])
            crops = fixed_grid_crops(transformed_img, nodes=17)
            crops = [self.processor(c) for c in crops]
            assert len(crops) == 17, f"Number of crops should be 17 but found {len(crops)}"
         
            
           
           
                
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)
        if(self.processor != None):
            return output['img'], output['label'], crops
        else:
            return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img






def collate_fn(data):
    if len(data[0]) > 2:
        img, labels, all_crops = zip(*data)
        return torch.stack(img), torch.tensor(labels), torch.stack(all_crops)
    img, labels = zip(*data)
    return torch.stack(img), torch.tensor(labels)

def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None,
    processor=None,
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper


    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train, processor=processor),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0

    return data_loader
