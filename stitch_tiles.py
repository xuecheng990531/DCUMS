import os
import re
from PIL import Image

def stitch_tiles(input_dir, output_file):
    # Get all files in the directory
    files = os.listdir(input_dir)
    
    # Filter for token_r...c...val...png
    pattern = re.compile(r'token_r(\d+)_c(\d+)_val[\d\.]+\.png')
    
    tiles = []
    max_r = 0
    max_c = 0
    
    for f in files:
        match = pattern.match(f)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            tiles.append((r, c, f))
            max_r = max(max_r, r)
            max_c = max(max_c, c)
            
    if not tiles:
        print("No tiles found.")
        return

    # Assuming all tiles have the same size, get size from the first one
    first_tile_path = os.path.join(input_dir, tiles[0][2])
    with Image.open(first_tile_path) as img:
        tile_w, tile_h = img.size
        
    rows = max_r + 1
    cols = max_c + 1
    
    canvas_w = cols * tile_w
    canvas_h = rows * tile_h
    
    print(f"Creating canvas: {canvas_w}x{canvas_h} from {rows}x{cols} tiles of size {tile_w}x{tile_h}")
    
    canvas = Image.new('RGB', (canvas_w, canvas_h))
    
    for r, c, filename in tiles:
        tile_path = os.path.join(input_dir, filename)
        with Image.open(tile_path) as img:
            canvas.paste(img, (c * tile_w, r * tile_h))
            
    canvas.save(output_file)
    print(f"Saved stitched image to {output_file}")

if __name__ == "__main__":
    input_dir = '/icislab/volume1/lxc/DCUMS/visualizations_train/token_tiles'
    output_file = '/icislab/volume1/lxc/DCUMS/visualizations_train/reconstructed_image.png'
    stitch_tiles(input_dir, output_file)
