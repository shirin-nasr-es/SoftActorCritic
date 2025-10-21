"""
This script filters and saves non-redundant UAV images by comparing frame-to-frame histogram similarity scores, keeping only visually distinct frames.
"""
import os
import cv2
import time
import shutil

# ---------- calculate histogram similarity score ----------
def histogram_similarity_score(im1, im2):
    hist1 = cv2.calcHist([im1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist1[255, 255, 255] = 0 # Ignore all white pixels
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)

    hist2 = cv2.calcHist([im2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2[255, 255, 255] = 0
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Find the metric value
    metric_value = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return metric_value


# ---------- read data ----------
input_dir = "<path_to_input_images>"
output_dir = "<path_to_output_folder>"

os.makedirs(output_dir, exist_ok=True)
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

frame_num = 1
im1 = cv2.imread(os.path.join(input_dir, image_files[0]))
print(im1.shape)
# im1 = cv2.resize(im1,(1920, 1080))
# print(im1.shape)

for idx in range(len(image_files) - 1):
    im2 = cv2.imread(os.path.join(input_dir, image_files[idx + 1]))
    # im2 = cv2.resize(im2, (1920, 1080))
    hs_sim = histogram_similarity_score(im1, im2)
    if hs_sim <= 0.9:         # 0.75 (rgb)
        im1 = im2
        print('yes')
        frame_num += 1
        cv2.imwrite(os.path.join(output_dir, image_files[idx + 1]), im2)
    else:
        print('No')
