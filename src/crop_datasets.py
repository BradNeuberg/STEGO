from data import *
from modules import *
import os
from data import ContrastiveSegDataset
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms.functional import five_crop, _get_image_size, crop, to_pil_image
from tqdm import tqdm
from torch.utils.data import Dataset


def _random_crops(img, size, seed, n):
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, int):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    images = []
    for i in range(n):
        seed1 = hash((seed, i, 0))
        seed2 = hash((seed, i, 1))
        crop_height, crop_width = int(crop_height), int(crop_width)

        top = seed1 % (image_height - crop_height)
        left = seed2 % (image_width - crop_width)
        images.append(crop(img, top, left, crop_height, crop_width))

    return images


class RandomCropComputer(Dataset):

    def _get_size(self, img):
        if len(img.shape) == 3:
            return [int(img.shape[1] * self.crop_ratio), int(img.shape[2] * self.crop_ratio)]
        elif len(img.shape) == 2:
            return [int(img.shape[0] * self.crop_ratio), int(img.shape[1] * self.crop_ratio)]
        else:
            raise ValueError("Bad image shape {}".format(img.shape))

    def random_crops(self, i, img):
        return _random_crops(img, self._get_size(img), i, 5)

    def five_crops(self, i, img):
        return five_crop(img, self._get_size(img))

    def __init__(self, cfg, dataset_name, img_set, crop_type, crop_ratio, create_label_preview):
        self.pytorch_data_dir = cfg.pytorch_data_dir
        self.crop_ratio = crop_ratio
        self.save_dir = join(
            cfg.pytorch_data_dir, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.img_set = img_set
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.create_label_preview = create_label_preview

        self.img_dir = join(self.save_dir, "img", img_set)
        self.label_dir = join(self.save_dir, "label", img_set)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        self.label_cmap = None
        if dataset_name == "oldclouds":
            self.label_cmap = create_oldclouds_colormap()

        if crop_type == "random":
            cropper = lambda i, x: self.random_crops(i, x)
        elif crop_type == "five":
            cropper = lambda i, x: self.five_crops(i, x)
        else:
            raise ValueError('Unknown crop type {}'.format(crop_type))

        self.dataset = ContrastiveSegDataset(
            cfg.pytorch_data_dir,
            dataset_name,
            None,
            img_set,
            T.ToTensor(),
            ToTargetTensor(),
            cfg=cfg,
            num_neighbors=cfg.num_neighbors,
            pos_labels=False,
            pos_images=False,
            mask=False,
            aug_geometric_transform=None,
            aug_photometric_transform=None,
            extra_transform=cropper
        )

    def __getitem__(self, item):
        batch = self.dataset[item]
        imgs = batch['img']
        labels = batch['label']
        for crop_num, (img, label) in enumerate(zip(imgs, labels)):
            img_num = item * 5 + crop_num
            img_arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy().squeeze(-1)
            Image.fromarray(img_arr).save(join(self.img_dir, "{}.jpg".format(img_num)), 'JPEG')
            Image.fromarray(label_arr).save(join(self.label_dir, "{}.png".format(img_num)), 'PNG')

            if self.dataset_name == "oldclouds":
                if (label_arr.max()-1) >= 6:
                    print(f"Invalid label class count for item {item}, max: {label_arr.max()}")

            if self.create_label_preview:
                # Convert this to a "nice" image with an actual color
                # per class.
                preview_label = np.zeros((3, label_arr.shape[0], label_arr.shape[1]),
                                         dtype=np.uint8)
                preview_label = self.label_cmap[label_arr-1].astype(np.uint8)
                preview_label = to_pil_image(preview_label)
                preview_filename = join(self.label_dir, "{}_preview.png".format(img_num))
                preview_label.save(preview_filename, "PNG")

        return True

    def __len__(self):
        return len(self.dataset)


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=0, workers=True)

    # dataset_names = ["cityscapes", "cocostuff27"]
    # img_sets = ["train", "val"]
    # crop_types = ["five","random"]
    # crop_ratios = [.5, .7]

    #dataset_names = ["potsdam"]
    #dataset_names = ["cocostuff27"]
    dataset_names = ["oldclouds"]
    img_sets = ["train", "val"]
    crop_types = ["five"]
    crop_ratios = [.5]

    for crop_ratio in crop_ratios:
        for crop_type in crop_types:
            for dataset_name in dataset_names:
                for img_set in img_sets:
                    dataset = RandomCropComputer(cfg, dataset_name, img_set, crop_type, crop_ratio,
                                                 create_label_preview=True)
                    loader = DataLoader(dataset, 1, shuffle=False, num_workers=cfg.num_workers, collate_fn=lambda l: l)
                    for _ in tqdm(loader):
                        pass


if __name__ == "__main__":
    prep_args()
    my_app()
