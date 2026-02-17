import os
import multiprocessing as mp
import shutil
import sklearn
from dataclasses import dataclass
from pathlib import Path
import sklearn.model_selection
import yaml
import cv2
import numpy as np
import tqdm
from ultralytics import YOLO

from data import IMAGES

TRAIN_DATA = Path("data/train")
VAL_DATA = Path("data/val")




@dataclass
class ImageGenerator:
    fg_images: list[Path]
    bg_images: list[Path]

    @classmethod
    def from_folder(cls, bg_folder: Path, fg_folder: Path):
        bg_images = list(bg_folder.glob("*.jpg"))
        fg_images = list(fg_folder.glob("*.jpg"))
        return ImageGenerator(fg_images=fg_images, bg_images=bg_images)

    def generate_dataset(self, image_list: list[Path], dest_folder: Path):
        images = Path(dest_folder / "images")
        images.mkdir(exist_ok=True, parents=True)

        labels = Path(dest_folder / "labels")
        labels.mkdir(exist_ok=True, parents=True)


        r = lambda: np.random.randint(0, len(self.bg_images) - 1)
        N = len(image_list)
        random_bgs = [self.bg_images[r()] for _ in range(N)]

        with mp.Pool(
            os.cpu_count(),
        ) as pool:
            data = list(zip(image_list, random_bgs))
            size = 100
            for chunk in tqdm.tqdm(
                np.split(data, np.arange(size, len(data), size)),
                desc="Generating data",
                total=len(data) // size + 1,
            ):
                outputs = pool.starmap(self.generate_homography, chunk)

                for data in outputs:
                    if data:
                        img, gt_data = data
                        cv2.imwrite(str(images / f"{gt_data['img_id']}.jpg"), img)

                        with open(str(labels / f"{gt_data['img_id']}.txt"), "w") as f:
                            f.write(self.yolo_label(gt_data))

    def generate_homography(
        self, painting_path: Path, background_path: Path
    ) -> tuple[np.ndarray, dict]:
        # 1. Load the painting
        painting = cv2.imread(str(painting_path))
        bg = cv2.imread(str(background_path))
        img_id = painting_path.stem
        if painting is None:
            return
        h_p, w_p = painting.shape[:2]

        bg_h, bg_w, _ = bg.shape
        # Optional: Add some random noise to the background
        noise = np.random.randint(0, 30, (bg_h, bg_w, 3), dtype=np.int16)
        x1, x2, x3, x4 = np.random.normal(0, bg_w // 20, 4)
        y1, y2, y3, y4 = np.random.normal(0, bg_h // 20, 4)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 3. Define corners for Homography
        src_coords = np.float32([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]])

        # Generate distorted destination points
        dst_coords = np.array(
            [
                [bg_w // 4 + x1, bg_h // 4 + y1],
                [bg_w - bg_w // 4 + x2, bg_h // 4 + y2],
                [bg_w - bg_w // 4 + x3, bg_h - bg_h // 4 + y3],
                [bg_w // 4 + x4, bg_h - bg_h // 4 + y4],
            ],
            dtype=np.float32,
        )

        # 4. Homography Warp
        M = cv2.getPerspectiveTransform(src_coords, dst_coords)
        warped_painting = cv2.warpPerspective(painting, M, (bg_w, bg_h))

        # 5. Blend using a mask
        mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_coords.astype(int), 255)

        # Place warped painting onto random color background
        bg[mask > 0] = warped_painting[mask > 0]

        # 6. Prepare JSON GT
        gt_data = {
            "img_id": img_id,
            "dimensions": {"width": bg_w, "height": bg_h},
            "obb_coords": dst_coords.tolist(),
        }
        return bg, gt_data

    def yolo_label(self, gt_data, class_id=0):
        # 1. Setup paths
        w = gt_data["dimensions"]["width"]
        h = gt_data["dimensions"]["height"]
        coords = gt_data["obb_coords"]  # Expected: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        # 2. Normalize and Flatten
        normalized_coords = []
        for pt in coords:
            norm_x = pt[0] / w
            norm_y = pt[1] / h
            normalized_coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

        # 3. Create the YOLO line
        label_line = [class_id] + normalized_coords

        return " ".join(map(str, label_line)) + "\n"


def generate_data(delete_existing:bool = False):
    if delete_existing:
        if TRAIN_DATA.exists():
            shutil.rmtree(TRAIN_DATA)
        TRAIN_DATA.mkdir(exist_ok=True, parents=True)

        if VAL_DATA.exists():
            shutil.rmtree(VAL_DATA)
        VAL_DATA.mkdir(exist_ok=True, parents=True)


    gen = ImageGenerator.from_folder(
        bg_folder=Path.home() / "data/como_lake", fg_folder=IMAGES
    )
    Path("test").mkdir(exist_ok=True, parents=True)
    train, test = sklearn.model_selection.train_test_split(gen.fg_images, test_size=0.1, random_state=42)
    gen.generate_dataset(train, TRAIN_DATA)
    gen.generate_dataset(test, VAL_DATA)


def generate_yolo_yaml(yaml_path: Path):
    """
    Generates a YOLO data configuration YAML file.
    """
    # 1. Prepare the data structure
    data = {
        # 'path': TRAIN_DATA.parent.as_posix(), # Dataset root
        'train': TRAIN_DATA.resolve().as_posix(),
        'val': VAL_DATA.resolve().as_posix(),
        'nc': 1, # Number of classes
        'names': ["painting"]  # List of class names
    }
    
    # 2. Write to YAML file
    with open(yaml_path, 'w') as f:
        # sort_keys=False preserves the order of your dictionary
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    



def train():
    shutil.rmtree("runs", ignore_errors=True)
    model = YOLO("yolov8n-obb.pt")
    generate_yolo_yaml(Path("cfg.yaml"))
    model.train(
        data="cfg.yaml", 
        epochs=5,
        imgsz=640,
        batch=32,
        device=0,  # Use 'cpu' if no GPU available
    )
    Path("yolo").mkdir(exist_ok=True, parents=True)
    shutil.copy("runs/obb/train/weights/best.pt", "yolo/model.pt")
    


def inference():
    # 1. Load your custom trained OBB model
    model = YOLO("yolo/model.pt")

    # 2. Run inference on a test image
    model.predict(source="test_images/earring.jpg", save=True, conf=0.8)


generate_data(delete_existing=True)
train()
inference()
