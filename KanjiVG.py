from lxml import etree
from typing import Any, Optional
from PIL import Image, ImageFilter
import numpy as np
import os, io, copy, cairosvg

# Hexidecimal color values
black_color = "#000000"
white_color = "#FFFFFF"

# Path of KanjiVG svg directory
KanjiVG_dir = "C:\\Users\\josia\\Kanji-Recognition\\kanjivg\\kanji"

class HarchImage:

    """
    This class represents a hierarchal image object. 
    It contains a base image `PIL.Image.Image` object and component `KanjiVG.HarchImage` objects.
    To create :py:class:`KanjiVG.HarchImage` objects, use the :py:class:`KanjiVG.HarchImageBuilder` builder class.

    * :py:func:`KanjiVG.HarchImage.GetBaseImage`
        Returns an `PIL.Image.Image` object.
        The base image of the hierarchy.
    * :py:func:`KanjiVG.HarchImage.GetComponentImages`
        Returns an `list` object containing `PIL.Image.Image` objects.
        The base images of all component `KanjiVG.HarchImage` objects.
    * :py:func:`KanjiVG.HarchImage.GetComponents`
        Returns an `list` object containing component `KanjiVG.HarchImage` objects.
    * :py:func:`KanjiVG.HarchImage.GetProperty`
        Returns the value of the property associated with the key parameter `key` in the internal dictionary.
        Returns None if the property does not exist in the dictionary.
    """

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
    
    def GetComponents(self):
        return self.components
    
    def GetProperty(self, key: str):
        if key in self.properties.keys():
            return self.properties[key]
        else: # Property DNE
            return None

class HarchImageBuilder:

    """
    This class is a builder for the `KanjiVG.HarchImage` class.

    * :py:func:`KanjiVG.HarchImageBuilder.SetBase`
        Sets the base image of the `KanjiVG.HarchImage` object to parameter `base`.
    * :py:func:`KanjiVG.HarchImageBuilder.SetComponents`
        Sets the components `list` object to parameter `components`. 
    * :py:func:`KanjiVG.HarchImageBuilder.AddComponent`
        Appends parameter `component` to the `list` object of components.
    * :py:func:`KanjiVG.HarchImageBuilder.AddComponents`
        Appends all item in parameter `components` to the `list` object of components.
    * :py:func:`KanjiVG.HarchImageBuilder.SetProperties`
        Sets the properties `dict` object to the parameter `properties`.
    * :py:func:`KanjiVG.HarchImageBuilder.SetProperty`
        Adds a property to the properties `dict` object.
        parameter `key` is property key and parameter `value` is property value.
    * :py:func:`KanjiVG.HarchImageBuilder.Build`
        Creates and returns an `KanjiVG.HarchImage` object using the set internal builder variables.
    """

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

def RasterizeSvg(svg: str, g_blur_amount: float, bw_threshold: int, output_size: tuple, downsampling_scale: int) -> Image.Image:

    """
    Rasterizes an SVG to a PIL Image

    PIL Image mode is "L" (8-bit grayscale).
    Output Image pixels are black/white (0/255).

    :param svg: xml format svg encoded to a string
    :param g_blur_amount: The standard deviation of the gaussian kernel.
        Used for gaussian blur before downsampling.
    :param bw_threshold: The pixel threshold used for binarization.
        Restricted to [0, 255].
        Pixel values greater will be set to white (255).
    :param output_size: The output size of the final processed Image
        (x, y)
    :param downsampling_scale: The factor of the output size of which svg rasterization occurs.
        Downsampling occurs post-rasterization to meet output size
    :returns: An :py:class:`~PIL.Image.Image` object.
    """

    # Rasterize the svg to png format using cairosvg (larger scale for later downsampling)
    png_image_data = cairosvg.svg2png(bytestring=svg, 
                                      output_width=output_size[0]*downsampling_scale,
                                      output_height=output_size[1]*downsampling_scale, 
                                      background_color=black_color)

    new_image = Image.open(io.BytesIO(png_image_data)).convert("L") # Load and convert to grayscale PIL.Image.Image format
    new_image = new_image.filter(ImageFilter.GaussianBlur(g_blur_amount)) # Apply gaussian blur
    new_image = new_image.resize(output_size, Image.LANCZOS) # Downsample

    return new_image.point(lambda p: 255 if p > bw_threshold else 0) # Binarize using threshold and return

def MakeCharacterHarchy(filename: str, groupings: dict[str, int]) -> HarchImage:

    """
    Creates an `KanjiVG.HarchImage` from an svg file.
    Components are represented as stroke groupings. Components of components are individual strokes.
    The base images at each level are rasterized images of each representation.
    Only individual strokes are rasterized, and higher-level base images are combinations of rasterized 
    component base images.
    (stroke types not included in dictionary parameter `groupings` are rasterized and handled as independent groups)

    If svg is non stroke-based, defines base image as full rasterization of svg.

    :param filename: The full file path of the svg file
    :param groupings: The dictionary of stroke types and group index
    :returns: An `KanjiVG.HarchImage` object.
    """

    # Safety check groupings - every group index from max index to min index exists
    max_group_idx = max(groupings.values())
    min_group_idx = min(groupings.values())
    if max_group_idx - min_group_idx + 1 != len(set(groupings.values())):
        return None

    # Builder for character hierarchy
    character_builder = HarchImageBuilder()
    character_builder.SetProperty("type", "character")

    # Get labels
    base_name = os.path.basename(filename)[:5].split('.')[0]
    ucode = int(base_name[:5], 16)
    character_builder.SetProperty("character-name", base_name)
    character_builder.SetProperty("unicode-value", ucode)

    # SVG modification
    new_stroke_color = white_color
    new_stroke_width = 2

    # Downscaling settings
    bw_threshold = 15
    g_blur_amount = 0.5
    output_size = (64, 64)
    downsampling_scale = 4

    # Loading SVG file
    tree = etree.parse(filename)
    root = tree.getroot()
    ns_key = "svg"
    ns = {ns_key: "http://www.w3.org/2000/svg"}

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
    group_style = None
    for elem in root.xpath(".//svg:g[starts-with(@id, 'kvg:StrokePaths_')]", namespaces=ns):
        if group_style is not None:
            break
        style = elem.attrib.get("style", "")
        if "stroke:" in style and new_stroke_color is not None:
            style = style.replace(f"stroke:{black_color}", f"stroke:{new_stroke_color}")
            style = style.replace("stroke:black", f"stroke:{new_stroke_color}")
            elem.attrib["style"] = style
            group_style = style
        if "stroke-width:" in style and new_stroke_width is not None:
            style = style.replace("stroke-width:3", f"stroke-width:{new_stroke_width}")
            elem.attrib["style"] = style
            group_style = style
            
    # Remove stroke numbers from SVG
    targets = root.xpath(".//svg:g[starts-with(@id, 'kvg:StrokeNumbers_')]", namespaces=ns)
    stroke_numbers_total = 0
    for target in targets:
        stroke_numbers_total = len(target.xpath(".//svg:text", namespaces=ns))
        parent = target.getparent()
        if parent is not None:
            parent.remove(target)
    character_builder.SetProperty("stroke-numbers-total", stroke_numbers_total)

    # Get whole character image
    character_svg = etree.tostring(root, encoding="utf-8", xml_declaration=False)
    full_character_image = RasterizeSvg(character_svg, g_blur_amount, bw_threshold,
                                    output_size, downsampling_scale)
    character_builder.SetProperty("full-rasterized-image", full_character_image)

    # Get transformative properties of SVG
    viewBox = root.attrib.get("viewBox")
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    character_builder.SetProperty("svg-viewBox", viewBox)
    character_builder.SetProperty("svg-width", width)
    character_builder.SetProperty("svg-height", height)

    # Create builders for groupings
    group_builders = [HarchImageBuilder() for _ in set(groupings.values())]

    # Create svgs and images for individual strokes
    for path in root.xpath(".//svg:path", namespaces=ns):
        type_string = path.attrib.get("{http://kanjivg.tagaini.net}type")
        if type_string is None:
            parent = path.getparent()
            if parent is not None:
                parent.remove(path)
            continue
        
        # Builder for individual strokes
        stroke_builder = HarchImageBuilder()
        stroke_builder.SetProperty("type", "stroke")

        # Find stroke type in groupings (set to whole type string if DNE)
        stroke_type = type_string
        for i in range(len(type_string)-1, -1, -1):
            if type_string[i] in groupings:
                stroke_type = type_string[i]
                break
        stroke_builder.SetProperty("stroke-type", stroke_type)

        # Create new SVG for stroke
        new_root = etree.Element(ns_key, nsmap={None: ns[ns_key]})
        if viewBox: new_root.attrib["viewBox"] = viewBox
        if width: new_root.attrib["width"] = width
        if height: new_root.attrib["height"] = height
        new_group = etree.Element("g")
        new_group.attrib["style"] = group_style
        new_group.attrib["id"] = f'kvg:StrokePaths_{base_name}'
        new_group.append(copy.deepcopy(path))
        new_root.append(copy.deepcopy(new_group))

        # Rasterize stroke image and add to builder
        stroke_svg = etree.tostring(new_root, encoding="utf-8", xml_declaration=False)
        stroke_image = RasterizeSvg(stroke_svg, g_blur_amount, bw_threshold,
                                    output_size, downsampling_scale)
        stroke_builder.SetBase(stroke_image)

        # Build stroke object and add to group (or character if no valid group)
        if stroke_type in groupings:
            group_builders[groupings[stroke_type]].AddComponent(stroke_builder.build())
        else:
            character_builder.AddComponent(stroke_builder.build())
    
    # Create group objects
    for idx, group_builder in enumerate(group_builders):
        group_builder.SetProperty("type", "stroke-group")
        if group_builder.components is not None:
            stroke_images = [component.GetBaseImage() for component in group_builder.components]
            group_image = CombineImages(stroke_images, output_size)
            group_builder.SetBase(group_image)
        else: # Get black image if DNE
            group_image = CombineImages(None, output_size)
            group_builder.SetProperty("blank", True)
            group_builder.SetBase(group_image)

        group_builder.SetProperty("group-index", idx)
        group_types = [key for key, value in groupings.items() if value == idx]
        group_builder.SetProperty("group-types", group_types)
        character_builder.AddComponent(group_builder.build())

    # No strokes in svg (Non-stroke based character)
    if [img for img in character_builder.components if img.GetProperty("blank") is None] == []:
        character_builder.SetBase(full_character_image)
        character_builder.SetComponents(None)
        return character_builder.build()

    # Create combined image and build character
    group_images = [
        component.GetBaseImage() 
        for component in character_builder.components 
        if component.GetBaseImage() is not None
        ]
    character_image = CombineImages(group_images, output_size)
    character_builder.SetBase(character_image)
    return character_builder.build()

def CombineImagesNp(images: list[Image.Image], size: tuple) -> np.ndarray:

    """
    Combines all non-zero pixels of `PIL.Image.Image` objects in `list` object.
    All images in the lsit must have be the same size.

    :param images: The `list` object of `PIL.Image.Image` object to combine.
    :param size: The size of the output image. (x, y)
    :returns: An `numpy.ndarray` object which is a combination of input images.
    """

    # Create new PIL image & convert to NP array
    arr_combined = np.array(Image.new("L", size, 0))
    
    # Return black image if None
    if images is None:
        return arr_combined

    # Merge stroke images
    for img in images:
        arr_img = np.array(img) # Convert stroke image to NP array
        mask = arr_img > 0 # Create mask from white pixels
        arr_combined[mask & (arr_combined == 0)] = 255 # Add white pixels to combined image
    
    # Convert back to PIL image
    return arr_combined

def CombineImages(images: list[Image.Image], size: tuple):

    """
    Function wrapper for `KanjiVG.CombineImagesNp`.
    returns the combination of images converted to an `PIL.Image.Image` object.

    :param images: The `list` object of `PIL.Image.Image` object to combine.
    :param size: The size of the output image. (x, y)
    :returns: An `PIL.Image.Image` object which is a combination of input images.
    """

    return Image.fromarray(CombineImagesNp(images, size))