from lxml import etree
import torch

Groupings = {
    "㇀": 0,
    "㇁": 3,
    "㇂": 5,
    "㇃": 5,
    "㇄": 2,
    "㇅": 7,
    "㇆": 7,
    "㇇": 8,
    "㇈": 6,
    "㇉": 9,
    "㇋": 9,
    "㇏": 4,
    "㇐": 1,
    "㇑": 2,
    "㇒": 3,
    "㇓": 3,
    "㇔": 0,
    "㇕": 7,
    "㇖": 1,
    "㇗": 2,
    "㇙": 2,
    "㇚": 2,
    "㇛": 9,
    "㇜": 9,
    "㇞": 9,
    "㇟": 9,
    "㇡": 9,        
}

def save_as_image(svg):

    return None


def format_svg(filename: str):
    
    # Open SVG file
    tree = etree.parse(filename)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Remove stroke numbers from SVG
    targets = root.xpath(".//svg:g[starts-with(@id, 'kvg:StrokeNumbers_')]", namespaces=ns)
    for target in targets:
        parent = target.getparent()
        if parent is not None:
            parent.remove(target)

    # Convert stroke color to white
    for elem in root.xpath(".//svg:g | .//svg:path", namespaces=ns):
        style = elem.attrib.get("style", "")
        if "stroke:" in style:
            style = style.replace("stroke:#000000", "stroke:#FFFFFF")
            style = style.replace("stroke:black", "stroke:#FFFFFF")
            elem.attrib["style"] = style

    
    
    tree.write("mod.svg")

format_svg("kanjivg\\kanji\\0f9b2.svg")
