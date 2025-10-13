from lxml import etree
import cairosvg
import io
from PIL import Image, ImageFilter, ImageChops
import os
import numpy as np
from scipy.ndimage import label
from typing import Any, Optional

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

# Holds an image and a hierarchy of sub-images 
class HarchImage:
    def __init__(
        self,
        base: Image.Image,
        components: Optional[list["HarchImage"]] = None,
        properties: Optional[dict[str, Any]] = None,
    ):
        self.base = base
        self.components = components
        self.properties = properties

    def GetBaseImage(self):
        return self.base
    
    def GetComponentImages(self):
        return [component.base for component in self.components]

# Builder for the hierarchial image
class HarchImageBuilder:
    def __init__(self):
        self.base = None
        self.components = None
        self.properties = None
    
    def SetBase(self, base: Image.Image):
        self.base = base
        return self
    
    def SetComponents(self, components: list["HarchImage"]):
        self.components = components
        return self
    
    def AddComponent(self, component: HarchImage):
        self.components = (self.components or []) + [component]
        return self
    
    def AddComponents(self, components: list["HarchImage"]):
        self.components = (self.components or []) + components
        return self
    
    def SetProperties(self, properties: dict[str, Any]):
        self.properties = properties
        return self
    
    def SetProperty(self, key: str, value):
        if self.properties is None:
            self.properties = {key : value}
            return self
        self.properties[key] = value
        return self
            
    def build(self):
        return HarchImage(
            base = self.base,
            components = self.components,
            properties = self.properties
        )

def rasterize_svg(svg, g_blur_amount: float, bw_threshold: int, output_size: tuple, downsampling_scale: int):

    png_image_data = cairosvg.svg2png(bytestring=svg, 
                                      output_width=output_size[0]*downsampling_scale,
                                      output_height=output_size[1]*downsampling_scale, 
                                      background_color="#00000")

    new_image = Image.open(io.BytesIO(png_image_data)).convert("L")
    new_image = new_image.filter(ImageFilter.GaussianBlur(g_blur_amount))
    new_image = new_image.resize(output_size, Image.LANCZOS)

    return new_image.point(lambda p: 255 if p > bw_threshold else 0)

def MakeCharacterHarchy(filename: str, groupings: dict[str, int]):

    # SVG modification
    new_stroke_color = "#FFFFFF"
    new_stroke_width = 2

    # Downscaling settings
    bw_threshold = 15
    g_blur_amount = 0.5
    output_size = (64, 64)
    downsampling_scale = 4

    # Loading SVG file
    tree = etree.parse(filename)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Builder for character hierarchy
    character_builder = HarchImageBuilder()

    # Set basic properties
    character_builder.SetProperty("filename", os.path.basename(filename))
    character_builder.SetProperty("groupings", groupings)
    character_builder.SetProperty("new-stroke-color", new_stroke_color)
    character_builder.SetProperty("new-stroke-width", new_stroke_width)
    character_builder.SetProperty("bw-threshold", bw_threshold)
    character_builder.SetProperty("g-blur-amount", g_blur_amount)
    character_builder.SetProperty("output-size", output_size)
    character_builder.SetProperty("downsampling-scale", downsampling_scale)

    # Set new stroke color and width
    for elem in root.xpath(".//svg:g | .//svg:path", namespaces=ns):
        style = elem.attrib.get("style", "")
        if "stroke:" in style and new_stroke_color is not None:
            style = style.replace("stroke:#000000", f"stroke:{new_stroke_color}")
            style = style.replace("stroke:black", f"stroke:{new_stroke_color}")
            elem.attrib["style"] = style
        if "stroke-width:" in style and new_stroke_width is not None:
            style = style.replace("stroke-width:3", f"stroke-width:{new_stroke_width}")
            elem.attrib["style"] = style

    # Remove stroke numbers from SVG
    stroke_numbers_total = 0
    targets = root.xpath(".//svg:g[starts-with(@id, 'kvg:StrokeNumbers_')]", namespaces=ns)
    for target in targets:
        parent = target.getparent()
        if parent is not None:
            parent.remove(target)
            stroke_numbers_total += 1
    character_builder.SetProperty("stroke-numbers-total", stroke_numbers_total)

    # Get whole character image
    character_svg = etree.tostring(root, encoding="utf-8", xml_declaration=False)
    full_character_image = rasterize_svg(character_svg, g_blur_amount, bw_threshold,
                                    output_size, downsampling_scale)
    character_builder.SetProperty("full-rasterized-image", full_character_image)

    # Get transformative properties of SVG
    viewBox = root.attrib.get("viewbox")
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    character_builder.SetProperty("svg-viewBox", viewBox)
    character_builder.SetProperty("svg-width", width)
    character_builder.SetProperty("svg-height", height)

    # Create builders for groupings
    group_builders = [HarchImageBuilder()] * len(groupings.values())

    # Remove tags without stroke type attribute
    for path in root.xpath(".//svg:path", namespaces=ns):
        type_string = path.attrib.get("{http://kanjivg.tagaini.net}type")
        if type_string is None:
            parent = path.getparent()
            if parent is not None:
                parent.remove(path)
            continue
        
        # Builder for individual strokes
        stroke_builder = HarchImageBuilder()

        # Find stroke type in groupings (set to whole type string if DNE)
        stroke_type = type_string
        for i in range(start=len(type_string)-1, stop=-1, step=-1):
            if type_string[i] in groupings:
                stroke_type = type_string[i]
                break
        stroke_builder.SetProperty("stroke-type", stroke_type)
        
        # Create new SVG for stroke
        new_root = etree.Element("{" + f"{ns["svg"]}" + "}svg", xmlns={None: ns["svg"]})
        if viewBox: new_root.attrib["viewBox"] = viewBox
        if width: new_root.attrib["width"] = width
        if height: new_root.attrib["height"] = height
        new_group = etree.SubElement(new_root, "{" + f"{ns["svg"]}" + "}g", id="kvg:StrokePaths_Separate")
        new_group.append(path)

        # Rasterize stroke image and add to builder
        stroke_svg = etree.tostring(new_root, encoding="utf-8", xml_declaration=False)
        stroke_image = rasterize_svg(stroke_svg, g_blur_amount, bw_threshold,
                                    output_size, downsampling_scale)
        stroke_builder.SetBase(stroke_image)

        # Build stroke object and add to group (or character if no valid group)
        if stroke_type in groupings:
            group_builders[groupings[stroke_type]].AddComponent(stroke_builder.build())
        else:
            character_builder.AddComponent(stroke_builder.build())
    
    # Create group objects
    for idx, group_builder in enumerate(group_builders):
        stroke_images = [component.GetBaseImage() for component in group_builder.components]
        group_image = combine_images(stroke_images)
        group_builder.SetBase(group_image)
        group_builder.SetProperty("group-index", idx)
        group_types = [key for key, value in groupings.items() if value == idx]
        group_builder.SetProperty("group-types", group_types)
        character_builder.AddComponent(group_builder.build())

    # Create combined image and build character
    group_images = [component.GetBaseImage() for component in character_builder.components]
    character_image = combine_images(group_images)
    character_builder.SetBase(character_image)
    return character_builder.build()

def separate_similar_strokes(stroke_image):

    # Convert to np array
    stroke_image_arr = np.array(stroke_image)

    binary = (stroke_image_arr > 0).astype(np.uint8)
    labeled, num_features = label(binary)

    individual_stroke_images = []
    for i in range(1, num_features + 1):

        mask = (labeled == i).astype(np.uint8) * 255
        individual_stroke_image = Image.fromarray(mask, mode="L")
        individual_stroke_images.append(individual_stroke_image)
    return individual_stroke_images

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

def combine_images_np(strokes):

    # Create new PIL image & convert to NP array
    arr_combined = np.array(Image.new("L", strokes[0].size, 0))
    
    # Merge stroke images
    for img in strokes:
        arr_img = np.array(img) # Convert stroke image to NP array
        mask = arr_img > 0 # Create mask from white pixels
        arr_combined[mask & (arr_combined == 0)] = 255 # Add white pixels to combined image
    
    # Convert back to PIL image
    return arr_combined

def combine_images(strokes):
    return Image.fromarray(combine_images_np(strokes))

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
                character = MakeCharacterHarchy()

                # Check for overlapping stroke in group images
                separated_total_strokes = 0
                for img in stroke_images:
                    separated_total_strokes += len(separate_similar_strokes(img))
                if separated_total_strokes != num_strokes:
                    print(f"Mismatch Number of Strokes - Ex:{num_strokes} got: {separated_total_strokes}")
                    discarded.append(entry)
                    discarded_reason.append(f"Stroke number mismatch: Ex{num_strokes} got: {separated_total_strokes}")
                    continue

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