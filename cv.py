from pathlib import Path
import kornia_rs as K
import numpy as np
import cv2
from ultralytics import YOLO
import tqdm

from ultralytics import YOLO
import cv2

def run_yolo():
    # 1. Load the latest lightweight OBB model
    # 'yolo11n-obb.pt' is extremely fast and perfect for mobile/edge use
    model = YOLO("runs/obb/train/weights/best.pt")


    results = model.predict(source="earring.jpg", save=True, conf=0.3)
    idx = np.argmax([r.obb.conf.cpu().numpy() for r in results]).item()
    return results[idx].obb.xyxy



def straighten_painting_kornia_rs(image_path):
    # 1. Load the image using kornia_rs (loads as a KorniaImage object)
    # kornia_rs handles image decoding very efficiently
    image = K.read_image_any(image_path) 
    
    # Convert to numpy for the corner detection part (OpenCV is still best for this)
    
    # 2. Find Corners (Classical CV approach)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.imwrite( "edged.jpg", edged)
    painting_corners = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            painting_corners = approx.reshape(4, 2).astype(np.float32)
            break

    if painting_corners is None:
        return "Painting not detected."

    # 3. Define the destination points (Perfect Rectangle)
    dst_w, dst_h = 500, 700
    dst_pts = np.array([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]
    ], dtype=np.float32)

    # 4. Apply Homography / Warp using kornia_rs
    # kornia-rs provides a high-performance warping function
    # Note: We pass the image and the transformation parameters
    
    # In kornia-rs, we use the geometry module for warping
    # We calculate the perspective matrix first
    matrix = cv2.getPerspectiveTransform(painting_corners, dst_pts)
    
    # Perform the warp (kornia-rs warp_perspective is extremely fast)
    warped_image = K.warp_perspective(
        image, 
        matrix.flatten().tolist(), 
        (dst_h, dst_w), 
        interpolation="bilinear"
    )

    return warped_image



def get_image_hash(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Initialize the pHash algorithm
    phasher = cv2.img_hash.PHash_create()
    
    # Compute hash
    current_hash = phasher.compute(img)
    return current_hash

def are_duplicates(hash1, hash2, threshold=5):
    # Compare hashes using Hamming distance
    phasher = cv2.img_hash.PHash_create()
    distance = phasher.compare(hash1, hash2)
    
    # distance 0 = exact match
    # distance < 5-10 = very likely a duplicate/resized version
    return distance <= threshold


def compare_vectorized(query_hash, database_hashes, threshold=5):
    """
    query_hash: A single numpy array (1, 8) or (8,)
    database_hashes: A numpy array of shape (N, 8) 
    """
    # 1. XOR the query against the entire database
    # Broadcasting happens automatically here
    xor_result = np.bitwise_xor(query_hash, database_hashes)
    
    # 2. Count set bits (popcount) across the array
    # np.unpackbits converts bytes to individual 0s and 1s
    bits = np.unpackbits(xor_result.astype(np.uint8), axis=-1)
    distances = np.sum(bits, axis=-1)
    
    # 3. Return indices of images that are below the threshold
    return np.where(distances <= threshold)[0]

def find_duplicates():
    images = list(Path("rijks_images").glob("*.jpg"))
    hashes = [get_image_hash(m) for m in images]
    matches = compare_vectorized(hashes, hashes)
    print(matches)



    
run_yolo()
