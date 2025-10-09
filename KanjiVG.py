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
    bw_kanji_image = tmp_img.point(lambda p: 255 if p > px_threshold else 0, mode="1")

    stroke_images.append(bw_kanji_image)

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
                    exit(1)

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

        # Rasterize image
        stroke_svg = etree.tostring(temp_root, encoding="utf-8", xml_declaration=False)
        stroke_image = cairosvg.svg2png(bytestring=stroke_svg, output_width=256, output_height=256, background_color="#000000")
         
        tmp_img = Image.open(io.BytesIO(stroke_image)).convert("L")
        tmp_img = tmp_img.filter(ImageFilter.GaussianBlur(g_blur_factor))
        tmp_img = tmp_img.resize((64, 64), Image.LANCZOS)

        bw_stroke_image = tmp_img.point(lambda p: 255 if p > px_threshold else 0, mode="1")

        stroke_images.append(bw_stroke_image)

    return stroke_images

def verify_strokes(character, strokes):
    
    combined = Image.new("L", (64, 64), 0)

    for img in strokes:

        arr_comb = np.array(combined)
        arr_img = np.array(img)

        mask = np.any(arr_img > 0, axis=-1)

        arr_comb[mask] = arr_img[mask]
        combined = Image.fromarray(arr_comb)

    arr_comb = np.array(combined)
    arr_ref = np.array(character)

    reference = character.convert("L")

    # Compute absolute per-channel differences
    diff = np.abs(arr_comb.astype(int) - arr_ref.astype(int))

    # Mask where any channel differs
    diff_mask = np.any(diff > 0, axis=-1)

    num_diff = np.count_nonzero(diff_mask)
    if num_diff > 0:
        print(f"Images differ — {num_diff} pixels differ.")
        vis_diff = Image.merge("RGB", (reference, reference, reference))
        vis_arr = np.array(vis_diff)
        vis_arr[diff_mask] = [255, 0, 0]  # Mark differences in red
        Image.fromarray(vis_arr).save("temp-diff.png")
        Image.fromarray(arr_comb).save("temp-comb.png")
        reference.save("temp-ref.png")
        exit(1)

def generate_images():

    # Make output directory
    os.makedirs(Output_dir, exist_ok=True)
    total = 0

    for entry in os.listdir(KanjiVG_dir):
        full_path = os.path.join(KanjiVG_dir, entry)

        # Get every file
        if os.path.isfile(full_path):
            ucode = int(entry[:5], 16) # unicode value as integer
            bname = entry.split('.')[0] # file base name without extension
            
            if ucode >= ucode_min and ucode <= ucode_max: # Only characters in Kanji range
                images = rasterize_groupings(full_path)
                images[0].save(os.path.join(Output_dir, f'{bname}.png')) # Save full character

                # Save stroke images
                for idx in range(1, len(images)):
                    images[idx].save(os.path.join(Output_dir, f'{bname}-s{idx}.png'))
                total += 1

                print(entry)
                verify_strokes(images[0], images[1:])
                print(f'{total}/11370')

generate_images()

images = rasterize_groupings("kanjivg\\kanji\\05a35-Kaisho.svg")
for idx, image in enumerate(images):
    image.save(f"out_svgs\\img-{idx}.png")