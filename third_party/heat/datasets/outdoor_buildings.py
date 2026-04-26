import os
import numpy as np
import skimage
import cv2
from torchvision import transforms
from PIL import Image

from datasets.corners import CornersDataset
from datasets.data_utils import RandomBlur


class OutdoorBuildingDataset(CornersDataset):
    """
    Merged dataset logic.

    Training / validation / test:
    - GT is required
    - det_final is loaded if det_path is provided
    - behavior remains compatible with train.py

    Inference:
    - GT is optional
    - det_final is optional
    - if det_path is None, no det_final file is required
    """

    def __init__(self, data_path, det_path=None, phase='train', image_size=256, rand_aug=True, inference=False):
        super(OutdoorBuildingDataset, self).__init__(image_size, inference)

        self.data_path = data_path
        self.det_path = det_path
        self.phase = phase
        self.rand_aug = rand_aug
        self.image_size = image_size
        self.inference = inference

        blur_transform = RandomBlur()
        self.train_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.3),
            blur_transform
        ])

        self.training = (phase == 'train')
        self._data_names = self._load_data_names()

    def _load_data_names(self):
        if self.phase == 'train':
            datalistfile = os.path.join(self.data_path, 'train_list.txt')
            with open(datalistfile, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]

        if self.phase == 'valid':
            datalistfile = os.path.join(self.data_path, 'valid_list.txt')
            with open(datalistfile, 'r') as f:
                names = [line.strip() for line in f.readlines() if line.strip()]
            return names[:50]

        if self.phase == 'test':
            datalistfile = os.path.join(self.data_path, 'valid_list.txt')
            with open(datalistfile, 'r') as f:
                names = [line.strip() for line in f.readlines() if line.strip()]
            return names[50:]

        if self.phase == 'all':
            all_list_path = os.path.join(self.data_path, 'all_list.txt')
            if os.path.exists(all_list_path):
                with open(all_list_path, 'r') as f:
                    return [line.strip() for line in f.readlines() if line.strip()]

            rgb_dir = os.path.join(self.data_path, 'rgb')
            valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
            names = []
            for fn in sorted(os.listdir(rgb_dir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() in valid_exts:
                    names.append(stem)
            return names

        raise ValueError(f"Invalid phase: {self.phase}")

    def __len__(self):
        return len(self._data_names)

    def _find_image_path(self, data_name):
        rgb_dir = os.path.join(self.data_path, 'rgb')
        possible_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]

        for ext in possible_exts:
            candidate = os.path.join(rgb_dir, data_name + ext)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"No image found for '{data_name}' in {rgb_dir} with extensions {possible_exts}"
        )

    def _load_det_corners(self, data_name):
        if self.det_path is None:
            return np.zeros((0, 2), dtype=np.float32), ''

        det_path = os.path.join(self.det_path, data_name + '.npy')
        if not os.path.exists(det_path):
            return np.zeros((0, 2), dtype=np.float32), det_path

        det_corners = np.array(np.load(det_path, allow_pickle=True))
        if det_corners.ndim == 2 and det_corners.shape[1] == 2:
            det_corners = det_corners[:, ::-1]  # y,x -> x,y
        else:
            det_corners = np.zeros((0, 2), dtype=np.float32)

        return det_corners, det_path

    def __getitem__(self, idx):
        data_name = self._data_names[idx]
        print(f"🧪 Processing: {data_name}")

        annot_path = os.path.join(self.data_path, 'annot', data_name + '.npy')
        has_gt = os.path.exists(annot_path)

        if self.training and not has_gt:
            raise FileNotFoundError(f"Training requires GT annotation, but missing: {annot_path}")

        annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist() if has_gt else {}

        det_corners, det_path = self._load_det_corners(data_name)

        img_path = self._find_image_path(data_name)
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        orig_h, orig_w = rgb.shape[:2]

        # Important fix: only resize if needed
        if orig_h != self.image_size or orig_w != self.image_size:
            if has_gt:
                rgb, annot, det_corners = self.resize_data(rgb, annot, det_corners)
            else:
                rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                if det_corners.shape[0] > 0:
                    det_corners = det_corners.astype(np.float32).copy()
                    det_corners[:, 0] *= self.image_size / float(orig_w)
                    det_corners[:, 1] *= self.image_size / float(orig_h)

        if self.rand_aug and has_gt:
            image, annot, _, det_corners = self.random_aug_annot(rgb, annot, det_corners=det_corners)
        else:
            image = rgb

        rec_mat = None

        if has_gt and len(annot) > 0:
            corners = np.array(list(annot.keys()), dtype=np.float32)[:, [1, 0]]
        else:
            corners = np.zeros((0, 2), dtype=np.float32)

        if (not self.inference) and has_gt and len(corners) > 100:
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        if self.training:
            corners += np.random.normal(0, 0, size=corners.shape)
            pil_img = Image.fromarray(image)
            image = self.train_transform(pil_img)
            image = np.array(image)

        image = skimage.img_as_float(image)

        if corners.shape[0] > 0:
            sort_idx = np.lexsort(corners.T)
            corners = corners[sort_idx]
            corner_list = [(corners[i][1], corners[i][0]) for i in range(corners.shape[0])]
        else:
            corner_list = []

        raw_data = {
            'name': data_name,
            'corners': corner_list,
            'annot': annot if has_gt else {},
            'image': image,
            'rec_mat': rec_mat,
            'annot_path': annot_path if has_gt else '',
            'det_path': det_path,
            'img_path': img_path,
        }

        return self.process_data(raw_data)

    def random_aug_annot(self, img, annot, det_corners=None):
        img, annot, det_corners = self.random_flip(img, annot, det_corners)

        theta = np.random.randint(0, 360) / 360 * np.pi * 2
        r = self.image_size / 256
        origin = [127 * r, 127 * r]
        p1_new = [127 * r + 100 * np.sin(theta) * r, 127 * r - 100 * np.cos(theta) * r]
        p2_new = [127 * r + 100 * np.cos(theta) * r, 127 * r + 100 * np.sin(theta) * r]
        p1_old = [127 * r, 127 * r - 100 * r]
        p2_old = [127 * r + 100 * r, 127 * r]
        pts1 = np.array([origin, p1_old, p2_old]).astype(np.float32)
        pts2 = np.array([origin, p1_new, p2_new]).astype(np.float32)
        M_rot = cv2.getAffineTransform(pts1, pts2)

        all_corners = list(annot.keys())
        if det_corners is not None and det_corners.shape[0] > 0:
            for i in range(det_corners.shape[0]):
                all_corners.append(tuple(det_corners[i]))
        all_corners_ = np.array(all_corners)

        corner_mapping = dict()
        ones = np.ones([all_corners_.shape[0], 1])
        all_corners_ = np.concatenate([all_corners_, ones], axis=-1)
        aug_corners = np.matmul(M_rot, all_corners_.T).T

        for idx, corner in enumerate(all_corners):
            corner_mapping[corner] = aug_corners[idx]

        new_corners = np.array(list(corner_mapping.values()))
        if new_corners.min() <= 0 or new_corners.max() >= (self.image_size - 1):
            return img, annot, None, det_corners

        aug_annot = dict()
        for corner, connections in annot.items():
            new_corner = corner_mapping[corner]
            tuple_new_corner = tuple(new_corner)
            aug_annot[tuple_new_corner] = []
            for to_corner in connections:
                aug_annot[tuple_new_corner].append(corner_mapping[tuple(to_corner)])

        rows, cols, _ = img.shape
        new_img = cv2.warpAffine(img, M_rot, (cols, rows), borderValue=(255, 255, 255))

        y_start = (new_img.shape[0] - self.image_size) // 2
        x_start = (new_img.shape[1] - self.image_size) // 2
        aug_img = new_img[y_start:y_start + self.image_size, x_start:x_start + self.image_size, :]

        if det_corners is None or det_corners.shape[0] == 0:
            return aug_img, aug_annot, corner_mapping, None

        aug_det_corners = []
        for corner in det_corners:
            new_corner = corner_mapping[tuple(corner)]
            aug_det_corners.append(new_corner)
        aug_det_corners = np.array(aug_det_corners)

        return aug_img, aug_annot, corner_mapping, aug_det_corners


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from datasets.data_utils import collate_fn

    DATAPATH = './data/cities_dataset'
    DET_PATH = './data/det_final'
    dataset = OutdoorBuildingDataset(DATAPATH, DET_PATH, phase='train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for i, item in enumerate(dataloader):
        import pdb
        pdb.set_trace()
        print(item)