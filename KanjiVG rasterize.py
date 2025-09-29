from lxml import etree
from io import BytesIO
import cairosvg
from PIL import Image

Groupings = {
    "㇀": 0, "㇁": 3, "㇂": 5, "㇃": 5, "㇄": 2, "㇅": 7, "㇆": 7,
    "㇇": 8, "㇈": 6, "㇉": 9, "㇋": 9, "㇏": 4, "㇐": 1, "㇑": 2,
    "㇒": 3, "㇓": 3, "㇔": 0, "㇕": 7, "㇖": 1, "㇗": 2, "㇙": 2,
    "㇚": 2, "㇛": 9, "㇜": 9, "㇞": 9, "㇟": 9, "㇡": 9,
}

def rasterize_groupings(filename: str, size=(64, 64)):
    tree = etree.parse(filename)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Get all stroke <path> elements
    strokes = root.xpath(".//svg:path", namespaces=ns)

    # Collect images for each grouping index
    results = {}

    for group_idx in set(Groupings.values()):
        # Clone the tree
        temp_tree = etree.ElementTree(etree.fromstring(etree.tostring(root)))
        temp_root = temp_tree.getroot()

        # Remove strokes not belonging to this group
        for path in temp_root.xpath(".//svg:path", namespaces=ns):
            kvg_type = path.attrib.get("{http://kanjivg.tagaini.net}type")
            if kvg_type is None:
                parent = path.getparent()
                if parent is not None:
                    parent.remove(path)
                continue

            group = Groupings.get(kvg_type, None)
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

        # Convert strokes to white
        for elem in temp_root.xpath(".//svg:path | .//svg:g", namespaces=ns):
            style = elem.attrib.get("style", "")
            if "stroke:" in style:
                style = style.replace("stroke:#000000", "stroke:#FFFFFF")
                style = style.replace("stroke:black", "stroke:#FFFFFF")
                elem.attrib["style"] = style

        # Serialize modified SVG
        svg_data = etree.tostring(temp_root)

        # Rasterize with cairosvg (RGBA, transparent background)
        png_bytes = cairosvg.svg2png(bytestring=svg_data, output_width=size[0], output_height=size[1])

        # Load into Pillow
        img = Image.open(BytesIO(png_bytes)).convert("RGBA")

        # Apply black background
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        final_img = Image.alpha_composite(background, img).convert("RGB")

        results[group_idx] = final_img

    return results


# Example usage
images = rasterize_groupings("kanjivg\\kanji\\0f9b2.svg")
for idx, img in images.items():
    img.save(f"group_{idx}.png")  # Just for demo, you can keep them in memory
