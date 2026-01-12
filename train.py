import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tqdm
from ultralytics import YOLO


@dataclass
class ImageGenerator:
    fg_images: list[Path]
    bg_images: list[Path]
    
    @classmethod
    def from_folder(cls, bg_folder: Path, fg_folder:Path):
        bg_images = list(bg_folder.glob("*.jpg")) 
        fg_images = list(fg_folder.glob("*.jpg"))
        return ImageGenerator(fg_images=fg_images, bg_images=bg_images)

    def generate_dataset(self, n:int, data_type:str):
        shutil.rmtree("images", ignore_errors=True )
        shutil.rmtree("labels", ignore_errors=True )

        
        Path(f"images/{data_type}").mkdir(exist_ok=True, parents=True)
        Path(f"labels/{data_type}").mkdir(exist_ok=True, parents=True)
        iters = range(min(len(self.fg_images), n))
        for i in tqdm.tqdm(iters, desc=f"generating {data_type} data"):
            r = np.random.randint(0, len(self.bg_images)-1)
            data = self.generate_homography(self.fg_images[i], self.bg_images[r])
            if data:
                img, gt_data = data
                cv2.imwrite(f"images/{data_type}/{gt_data['img_id']}.jpg", img)

                with open(f"labels/{data_type}/{gt_data['img_id']}.txt", "w") as f:
                    labels = self.yolo_label(gt_data)
                    f.write(labels)


    def generate_homography(self, painting_path:Path, background_path:Path):
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
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 3. Define corners for Homography
        src_coords = np.float32([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]])

        # Generate distorted destination points
        margin = 100
        dst_coords = np.float32(
            [
                [random.uniform(margin, bg_w / 2), random.uniform(margin, bg_h / 2)],
                [random.uniform(bg_w / 2, bg_w - margin), random.uniform(margin, bg_h / 2)],
                [
                    random.uniform(bg_w / 2, bg_w - margin),
                    random.uniform(bg_h / 2, bg_h - margin),
                ],
                [random.uniform(margin, bg_w / 2), random.uniform(bg_h / 2, bg_h - margin)],
            ]
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




def train():
    # 1. Initialize the model (YOLOv8 or YOLOv11 OBB)
    shutil.rmtree("runs", ignore_errors=True )
    model = YOLO("yolov8n-obb.pt")
    gen = ImageGenerator.from_folder(Path.home()/ "data/como_lake", Path("rijks_images"))
    gen.generate_dataset(1000, "train")
    gen.generate_dataset(200, "val")


    # 2. Overwrite the trainer with our custom generator trainer
    # We use the 'train' method but pass our custom class via the 'trainer' argument
    model.train(
        data="cfg.yaml",  # Still needed for class names/paths
        epochs=4,
        imgsz=640,
        batch=32,
        device=0,  # Use 'cpu' if no GPU available
    )


def inference():
    # 1. Load your custom trained OBB model
    model = YOLO("runs/obb/train/weights/best.pt")

    # 2. Run inference on a test image
    results = model.predict(source="earring.jpg", save=True, conf=0.8)


train()
