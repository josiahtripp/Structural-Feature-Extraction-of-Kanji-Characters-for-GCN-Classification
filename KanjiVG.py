from lxml import etree
import cairosvg
import io
from PIL import Image, ImageFilter, ImageChops
import os
import numpy as np

Groupings = {
    "㇀": 0, "丶":0, "㇁": 3, "㇂": 5, "㇃": 5, "㇄": 2, "㇅": 7, "㇆": 7,
    "㇇": 8, "㇈": 6, "㇉": 9, "㇋": 9, "㇏": 4, "㇐": 1, "㇑": 2,
    "㇒": 3, "㇓": 3, "㇔": 0, "㇕": 7, "㇖": 1, "㇗": 2, "㇙": 2,
    "㇚": 2, "㇛": 9, "㇜": 9, "㇞": 9, "㇟": 9, "㇡": 9,
}

ucode_min = int("04e00", 16)
ucode_max = int("09fff", 16)

KanjiVG_dir = "C:\\Users\\josia\\Kanji-Recognition\\kanjivg\\kanji"
Output_dir = "C:\\Users\\josia\\Kanji-Recognition\\kanjivg-segmented"
Temporary_dir = "C:\\Users\\josia\\Kanji-Recognition\\kanjivg-tmp"

def rasterize_groupings(filename: str):

    px_threshold = 15
    g_blur_factor = 0.5

    tree = etree.parse(filename)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Collect images for each grouping index
    stroke_images = []

    # Convert stroke color to white
    for elem in root.xpath(".//svg:g | .//svg:path", namespaces=ns):
        style = elem.attrib.get("style", "")
        if "stroke:" in style:
            style = style.replace("stroke:#000000", "stroke:#FFFFFF")
            style = style.replace("stroke:black", "stroke:#FFFFFF")
            elem.attrib["style"] = style
        if "stroke-width:" in style:
            style = style.replace("stroke-width:3", "stroke-width:2")
            elem.attrib["style"] = style

    # Remove stroke numbers from SVG
    targets = root.xpath(".//svg:g[starts-with(@id, 'kvg:StrokeNumbers_')]", namespaces=ns)
    for target in targets:
        parent = target.getparent()
        if parent is not None:
            parent.remove(target)

    # Save whole character image
    whole_kanji_svg = etree.tostring(root, encoding="utf-8", xml_declaration=False)
    whole_kanji_image = cairosvg.svg2png(bytestring=whole_kanji_svg, output_width=256, output_height=256, background_color="#00000")

    tmp_img = Image.open(io.BytesIO(whole_kanji_image)).convert("L")
    tmp_img = tmp_img.filter(ImageFilter.GaussianBlur(g_blur_factor))
    tmp_img = tmp_img.resize((64, 64), Image.LANCZOS)
    character_image = tmp_img.point(lambda p: 255 if p > px_threshold else 0, mode="1")
    character_image = character_image.convert("L")

    viewBox = root.attrib.get("viewbox")
    width = root.attrib.get("width")
    height = root.attrib.get("height")

    # For all stroke groups
    for group_idx in set(Groupings.values()):

        # Clone the tree
        temp_tree = etree.ElementTree(etree.fromstring(etree.tostring(root)))
        temp_root = temp_tree.getroot()

        # Remove tags without stroke type attribute
        for path in temp_root.xpath(".//svg:path", namespaces=ns):
            kvg_type = path.attrib.get("{http://kanjivg.tagaini.net}type")
            if kvg_type is None:
                parent = path.getparent()
                if parent is not None:
                    parent.remove(path)
                continue

            # Remove stroke
            label_idx = len(kvg_type) - 1
            while kvg_type[label_idx] not in Groupings:
                label_idx -= 1
                if label_idx < 0:
                    print("Invalid Stroke Type:", kvg_type)
                    return None, kvg_type

            group = Groupings.get(kvg_type[label_idx], None)
            if group != group_idx:
                parent = path.getparent()
                if parent is not None:
                    parent.remove(path)

        # Remove empty <g> tags
        for g in temp_root.xpath(".//svg:g", namespaces=ns):
            if len(g) == 0:
                parent = g.getparent()
                if parent is not None:
                    parent.remove(g)

        if viewBox: temp_root.attrib["viewBox"] = viewBox
        if width: temp_root.attrib["width"] = width
        if height: temp_root.attrib["height"] = height

        # Rasterize image
        stroke_svg = etree.tostring(temp_root, encoding="utf-8", xml_declaration=False)
        stroke_image = cairosvg.svg2png(bytestring=stroke_svg, output_width=256, output_height=256, background_color="#000000", unsafe=True)
         
        tmp_img = Image.open(io.BytesIO(stroke_image)).convert("L")
        tmp_img = tmp_img.filter(ImageFilter.GaussianBlur(g_blur_factor))
        tmp_img = tmp_img.resize((64, 64), Image.LANCZOS)

        bw_stroke_image = tmp_img.point(lambda p: 255 if p > px_threshold else 0, mode="1")

        stroke_images.append(bw_stroke_image.convert("L"))

    return character_image, stroke_images

def get_num_strokes(filename: str):

    # Initialize
    num_strokes = 0

    # Open svg for reading
    tree = etree.parse(filename)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Count tags with stroke type attribute
    for path in root.xpath(".//svg:path", namespaces=ns):
        kvg_type = path.attrib.get("{http://kanjivg.tagaini.net}type")
        if kvg_type is None:
            parent = path.getparent()
            if parent is not None:
                parent.remove(path)
            continue
    
        # Increment stroke count
        num_strokes += 1
    return num_strokes

def combine_strokes_np(strokes):

    # Create new PIL image & convert to NP array
    arr_combined = np.array(Image.new("L", (64, 64), 0))
    
    # Merge stroke images
    for img in strokes:
        arr_img = np.array(img) # Convert stroke image to NP array
        mask = arr_img > 0 # Create mask from white pixels
        arr_combined[mask & (arr_combined == 0)] = 255 # Add white pixels to combined image
    
    # Convert back to PIL image
    return arr_combined

def combine_strokes(strokes):
    return Image.fromarray(combine_strokes_np(strokes))

def verify_strokes(character, strokes):

    # Combined strokes
    combined_arr = combine_strokes_np(strokes)

    # Convert character image
    character_arr = np.array(character)

    # Compute absolute per-channel differences
    diff = np.abs(combined_arr - character_arr)

    # Mask where any channel differs
    diff_mask = diff > 0

    # Number of different pixels
    num_diff = np.count_nonzero(diff_mask)

    # Images differ
    if num_diff > 0:

        # Output number of different pixels
        print(f"Images differ — {num_diff} pixels differ.")

        # Create temporary output directory
        os.makedirs(Temporary_dir, exist_ok=True)

        # Save temp image of difference higlighted in red
        vis_diff = Image.merge("RGB", (character, character, character))
        vis_diff_arr = np.array(vis_diff)
        vis_diff_arr[diff_mask] = [255, 0, 0]  # Mark differences in red
        vis_diff = Image.fromarray(vis_diff_arr)
        vis_diff.save(os.path.join(Temporary_dir, "v_strks() vis_diff.png"))

        # Save temp images of character and merged character
        character.save(os.path.join(Temporary_dir, "v_strks() character.png"))
        combined = Image.fromarray(combined_arr)
        combined.save(os.path.join(Temporary_dir, "v_strks() combined.png"))

        # Quit - Failed validation
        quit(1)

def verify_combined(original, combined, num_strokes):
    
    # Maximum allowed variation
    max_diff = num_strokes

    # Compute absolute per-channel differences
    original_arr = np.array(original)
    combined_arr = np.array(combined)

    diff = np.abs(original_arr- combined_arr)

    # Mask where any channel differs
    diff_mask = diff > 0

    # Number of different pixels
    num_diff = np.count_nonzero(diff_mask)

    # Images differ
    if num_diff > max_diff:

        # Output number of different pixels
        print(f"Combined | Original differ — {num_diff} pixels differ.")

        # Create temporary output directory
        os.makedirs(Temporary_dir, exist_ok=True)

        # Save temp image of difference higlighted in red
        vis_diff = Image.merge("RGB", (original, original, original))
        vis_diff_arr = np.array(vis_diff)
        vis_diff_arr[diff_mask] = [255, 0, 0]  # Mark differences in red
        vis_diff = Image.fromarray(vis_diff_arr)
        vis_diff.save(os.path.join(Temporary_dir, "v_comb() vis_diff.png"))

        # Save temp images of original and combined
        original.save(os.path.join(Temporary_dir, "v_comb() original.png"))
        combined.save(os.path.join(Temporary_dir, "v_comb() combined.png"))

        # Failed validation
        return num_diff
    return None

def generate_images():

    # Make output directory
    os.makedirs(Output_dir, exist_ok=True)
    discarded = []
    discarded_reason = []
    total_processed = 0
    total = 0

    # Tally up total number of files
    for entry in os.listdir(KanjiVG_dir):
        full_path = os.path.join(KanjiVG_dir, entry)
        if os.path.isfile(full_path):
            total += 1
    
    # Process Files
    for entry in os.listdir(KanjiVG_dir):
        full_path = os.path.join(KanjiVG_dir, entry)

        # Get every file
        if os.path.isfile(full_path):
            ucode = int(entry[:5], 16) # unicode value as integer
            bname = entry.split('.')[0] # file base name without extension
            
            if ucode >= ucode_min and ucode <= ucode_max: # Only characters in Kanji range

                print(entry)

                # Get rasterized images of strokes
                character_image, stroke_images = rasterize_groupings(full_path)

                if character_image is None:
                    print("Bad stroke type - discarding")
                    discarded.append(entry)
                    discarded_reason.append(f"Unknown Stroke: {stroke_images}")
                    continue

                # Get the number of strokes
                num_strokes = get_num_strokes(full_path)

                # Combine stroke images to create whole character
                combined_character_image = combine_strokes(stroke_images)

                # Check combined image
                comb_result = verify_combined(character_image, combined_character_image, num_strokes)
                if comb_result is not None:
                    print("Comb - Org char diff")
                    discarded.append(entry)
                    discarded_reason.append(f"comb/org diff - {comb_result}px")

                # Save full character
                combined_character_image.save(os.path.join(Output_dir, f'{bname}.png'))

                # Save stroke images
                for idx in range(len(stroke_images)):
                    stroke_images[idx].save(os.path.join(Output_dir, f'{bname}-s{idx}.png'))
 
                total_processed += 1
                print(f'\t| {total_processed}/{total} | {int(total_processed*100/total)}% Completed')
    
    print(f"\n--\nGenerated {total_processed}/{total} character/stroke-group pairs")
    print(f"Discared {len(discarded)} files:")
    for (ds , ds_rs) in zip(discarded, discarded_reason):
        print(f"\t{ds} | {ds_rs}")
    print(f"Total files generated: {11 * total_processed}")

generate_images()