import shutil, os, random

SOURCE_IMAGES_DIR = r"C:\EDP\WEEK4\Images"  # adjust the subfolder name below if needed

os.makedirs("samples", exist_ok=True)

all_images = os.listdir(SOURCE_IMAGES_DIR)
sample_pool = random.sample(all_images, 50)

for img in sample_pool:
    shutil.copy(os.path.join(SOURCE_IMAGES_DIR, img), os.path.join("samples", img))

print("Copied", len(sample_pool), "images into samples/")