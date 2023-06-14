from heic2png import HEIC2PNG
from pathlib import Path
from tqdm import tqdm
if __name__ == '__main__':
    for p in tqdm(Path('direction/direction_images/').glob('IMG_*.HEIC')):
        print(p)
        heic_img = HEIC2PNG(p)
        heic_img.save() # it'll show as `test.png`