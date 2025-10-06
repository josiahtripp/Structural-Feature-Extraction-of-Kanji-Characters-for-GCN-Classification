from lxml import etree
import cairosvg
import io
from PIL import Image, ImageOps, ImageFilter

Groupings = {
    "㇀": 0, "㇁": 3, "㇂": 5, "㇃": 5, "㇄": 2, "㇅": 7, "㇆": 7,
    "㇇": 8, "㇈": 6, "㇉": 9, "㇋": 9, "㇏": 4, "㇐": 1, "㇑": 2,
    "㇒": 3, "㇓": 3, "㇔": 0, "㇕": 7, "㇖": 1, "㇗": 2, "㇙": 2,
    "㇚": 2, "㇛": 9, "㇜": 9, "㇞": 9, "㇟": 9, "㇡": 9,
}

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
    whole_kanji_image = cairosvg.svg2png(bytestring=whole_kanji_svg, output_width=256, output_height=256, background_color="#000000")

    tmp_img = Image.open(io.BytesIO(whole_kanji_image)).convert("L")
    tmp_img = tmp_img.filter(ImageFilter.GaussianBlur(g_blur_factor))
    tmp_img = tmp_img.resize((64, 64), Image.LANCZOS)
    bw_kanji_image = tmp_img.point(lambda p: 255 if p > px_threshold else 0, mode="1")

    stroke_images.append(bw_kanji_image)

    # Get all stroke <path> elements
    strokes = root.xpath(".//svg:path", namespaces=ns)

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
            group = Groupings.get(kvg_type[len(kvg_type) - 1:], None)
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

images = rasterize_groupings("kanjivg\\kanji\\0f9b2.svg")
for idx, image in enumerate(images):
    image.save(f"out_svgs\\img-{idx}.png")