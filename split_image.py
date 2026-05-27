import os
from PIL import Image

def split_image(img_path, output_dir, tile_size=100):
    img = Image.open(img_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Grid calculation
    cols = img.width // tile_size
    rows = img.height // tile_size
    
    for row in range(rows):
        for col in range(cols):
            left = col * tile_size
            top = row * tile_size
            right = left + tile_size
            bottom = top + tile_size
            
            box = (left, top, right, bottom)
            tile = img.crop(box)
            tile.save(os.path.join(output_dir, f"tile_row{row}_col{col}.png"))
    
    print(f"Split {img_path} into {rows*cols} tiles in {output_dir}")

if __name__ == "__main__":
    split_image("visualizations_train/global_attention_block_0.png", "visualizations_train/block_0_tiles", tile_size=100)