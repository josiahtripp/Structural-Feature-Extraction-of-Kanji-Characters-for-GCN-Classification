from KanjiVG import MakeCharacterHarchy, HarchImage, KanjiVG_dir
import numpy as np
from PIL import Image
from scipy.ndimage import label
import os
import pickle
import torch
from torchvision import transforms

# Unicode range for Kanji characters
kanji_ucode_min = int("04e00", 16)
kanji_ucode_max = int("09fff", 16)

# Directories for storing processed images and temporary files
Output_dir = "C:\\Users\\josia\\Kanji-Recognition\\segmenter-training"
Temporary_dir = "C:\\Users\\josia\\Kanji-Recognition\\kanjivg-tmp"

# Groupings use for segmenting individual strokes into groups for the GAN stroke group tensor-field
groupings = {
    "㇀": 0, "丶":0, "㇁": 3, "㇂": 5, "㇃": 5, "㇄": 2, "㇅": 7, "㇆": 7,
    "㇇": 8, "㇈": 6, "㇉": 9, "㇋": 9, "㇏": 4, "㇐": 1, "㇑": 2,
    "㇒": 3, "㇓": 3, "㇔": 0, "㇕": 7, "㇖": 1, "㇗": 2, "㇙": 2,
    "㇚": 2, "㇛": 9, "㇜": 9, "㇞": 9, "㇟": 9, "㇡": 9,
}

# The number of stroke groups
num_stroke_groups = max(groupings.values()) - min(groupings.values()) + 1

def ImageIsBinary(img: Image.Image) -> bool:

    """
    Checks if an `PIL.Image.Image` object is binary (0/255)

    :param img: The image to check
    :returns: `True` if the image is binary and `False` if not.
    """

    img_arr = np.array(img)
    return np.all((img_arr == 0) | (img_arr == 255)) # Conditional if every pixel is either black or white (0 or 255)

def HarchImageIsBinary(img: HarchImage) -> bool:

    """
    Checks if the base image and all further component base image in an `KanjiVG.HarchImage` object are grayscale mode "L" and binary (0/255)

    :param img: The hierarchal image to check
    :returns: `True` if all images are binary and in the proper format and `False` if not.
    """

    if img is None: # No hierarchal image to check
        return True
    
    if img.GetBaseImage().mode != "L": # Not in correct mode
        return False
    
    if ImageIsBinary(img.GetBaseImage()) == False: # Base image is not binary
        return False
    
    # For all component hierarchal images
    components = img.GetComponents()
    if components is not None:
        for component in components:
            if HarchImageIsBinary(component) == False: # Component is not binary
                return False
            
    return True

def SeparatePixelGroups(stroke_image):

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

def basic_image_descriptor(img: Image.Image):

    descriptor = f"{img.size}"
    img_arr = np.array(img)
    num_white_pixels = np.count_nonzero(img_arr == 255)
    num_black_pixels = np.count_nonzero(img_arr == 0)

    if num_white_pixels + num_black_pixels == img_arr.size:
        descriptor += f' : B/W : {num_white_pixels} W : {num_black_pixels} B'
    else:
        descriptor += f' : NOT B/W : {num_white_pixels} W : {num_black_pixels} B : {img_arr.size - (num_white_pixels + num_black_pixels)} Other'

    return descriptor

def PrintProperties(CharacterImage: HarchImage) -> None:

    """
    Debugging function for printing properties of a hierarchal image.

    :param CharacterImage: The hierarchal image to print report of.
    """

    print("Properties:")
    for key, value in CharacterImage.properties.items():
        if value is None:
            print(f'\t{key}: None')
        elif type(value) in [str, int, float, tuple]:
            print(f'\t{key}: {value}')
        elif type(value) is dict:
            print(f'\t{key}: {len(value.values())} entries | {len(set(value.values()))} groups')
        elif type(value) is Image.Image:
            print(f'\t{key}: {basic_image_descriptor(value)}')
        else:
            print(f'\t{key}: Unknown Value Type \"{type(value)}\"')

    print("Base Image:")
    base_image = CharacterImage.GetBaseImage()
    if base_image is None:
        print("\tBase Image: None")
    elif type(base_image) is Image.Image:
        print(f"\tBase Image: {basic_image_descriptor(base_image)}")
    else:
        print(f"\tBase Image: Unknown Value Type \"{type(base_image)}\"")

    print("Components:")
    components = CharacterImage.GetComponents()
    for idx, component in enumerate(components):
        image_type = component.GetProperty("type")
        comp_image = component.GetBaseImage()
        strokes = component.GetComponents()
        group_types = component.GetProperty("group-types")
        group_index = component.GetProperty("group-index")

        if image_type is None:
            print(f'\t#{idx+1} - type: None')
        elif type(value) is str:
            print(f'\t#{idx+1} - type: {image_type}')
        else:
            print(f'\t#{idx+1} - type: Unknown Value Type \"{type(image_type)}\"')
        
        if group_types is None:
            print(f'\t\tgroup type: None')
        elif type(group_types) in [list[str], list]:
            print(f'\t\tgroup_type : {group_types}')
        else:
            print(f'\t\tgroup_type: Unknown Value Type \"{type(group_types)}\"')

        if group_index is None:
            print(f'\t\tgroup index: None')
        elif type(group_index) is int:
            print(f'\t\tgroup index : {group_index}')
        else:
            print(f'\t\tgroup index: Unknown Value Type \"{type(group_index)}\"')

        if comp_image is None:
            print("\t\tComponent Image: None")
        elif type(comp_image) is Image.Image:
            print(f"\t\tComponent Image: {basic_image_descriptor(comp_image)}")
        else:
            print(f"\t\tComponent Image: Unknown Value Type \"{type(comp_image)}\"")
        
        if strokes is None:
            print("\t\tStrokes: None")
        else:
            print(f"\t\tStrokes: {len(strokes)}")

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

def GetKanjiCharacters(regenerate=False) -> list[HarchImage]:

    """
    Generates Kanji character hierarchal images from the KnajiVG dataset

    :param regenerate: Regenerates character images from svgs if `True` and loads from file if `False`.
    :returns: An `list` object containing herarchal images of Kanji characters.
    """

    # Filename for list of valid character hierarchal images
    kanji_characters_filename = "kanji_characters.pkl"
    processing_info_filename = "kanji_characters_processing_info.txt"

    # Load from file if regen parameter is false
    if regenerate == False:

        # Load from file
        if os.path.exists(os.path.join(Output_dir, kanji_characters_filename)):
            with open(os.path.join(Output_dir, kanji_characters_filename), "rb") as file:

                kanji__characters = pickle.load(file) # Unpack file and return
                return kanji__characters
        # DNE
        return None

    # Get all files of SVGs
    files = [
        os.path.join(KanjiVG_dir, file) 
        for file in os.listdir(KanjiVG_dir) 
        if os.path.isfile(os.path.join(KanjiVG_dir, file))
        ]
    
    kanji_characters = []
    skipped_files = []

    # Get hierarchal images of svg files
    for idx, file in enumerate(files):

        # Only process svg if in Kanji range
        base_name = os.path.basename(file)[:5].split('.')[0] # Get the first 5 characters of the base file name
        ucode = int(base_name[:5], 16) # Get numerical unicode value
        if ucode <= kanji_ucode_max and ucode >= kanji_ucode_min:
            kanji_characters.append(MakeCharacterHarchy(file, groupings))

            ''' Debugging snippit for viewing base images during generation
            print_properties(characters[idx])
            plt.imshow(characters[idx].GetBaseImage(), cmap='gray')
            plt.title(os.path.basename(file))
            plt.axis("off")
            plt.show()
            '''
        else:
            skipped_files.append(file)

        # Print progress message
        progress_message = f"\rProcessed {idx+1} / {len(files)} - {int((idx+1)*100 / len(files))}% | {len(kanji_characters)} Kanji"
        progress_message += " " * (80 - len(progress_message))
        print(progress_message, end="", flush=True)

    # Print report of processing
    print(f"\nTotal: {len(kanji_characters)} Kanji characters from {len(files)} files")
    print(f'Skipped: {len(skipped_files)} files (Non-Kanji)')

    # Create output directory if it doesn't exist
    os.makedirs(Output_dir, exist_ok=True)
    
    # Save characters
    with open(os.path.join(Output_dir, kanji_characters_filename), "wb") as file:
        pickle.dump(kanji_characters, file)

    # Save processing info report
    with open(os.path.join(Output_dir, processing_info_filename), "w") as file:
        file.write("<--- Processing Information --->\n\n")
        file.write(f"Kanji character files processed: {len(kanji_characters)}\n")
        file.write(f"Non-kanji character files skipped: {len(skipped_files)}\n")
        file.write(f"Total Number of Files: {len(files)}\n\n")

        file.write("Skipped Files:\n") # All skipped files
        for f in skipped_files:
            file.write(f"\t{os.path.basename(f)}\n")

        file.write("\nKanji Characters Processed:\n") # All processed Kanji
        for character in kanji_characters:
            
            file.write("\tName: ") # Character name
            basename = character.GetProperty("character-name")
            if basename is None:
                file.write("(None)")
            else:
                file.write(basename)
            
            file.write(" | Filename: ") # Base filename
            filename = character.GetProperty("filename")
            if filename is None:
                file.write("(None)")
            else:
                file.write(os.path.basename(filename))

            file.write("\n")

        file.write("\nAll Files:\n") # All files (full path)
        for f in files:
            file.write(f"\t{f}\n")

def GetValidCharacters(regenerate=False, regenerate_kanji_characters=False) -> list[HarchImage]:

    """
    Validates and returns a list of kanji characters based on those produced by `SegmenterTraining.GetKanjiCharacters`.

    :param regenerate: Revalidates Kanji characters and regenrates list of valid characters if `True`, loads from fiel if `False`.
    :param regenerate_kanji_characters: Regenerates all images used in tensor generation if `True` and loads from save if `False`. Completely clean regen.
    :returns: An `list` object containing validated Kanji characters
    """

    # Filename for list of valid characters (also includes processing info)
    valid_characters_filename = "valid_characters.pkl"
    processing_info_filename = "valid_characters_processing_info.txt"

    # Load from file if regen parameter is false
    if regenerate == False and regenerate_kanji_characters == False:
        # Load from file
        if os.path.exists(os.path.join(Output_dir, valid_characters_filename)):
            with open(os.path.join(Output_dir, valid_characters_filename), "rb") as file:

                valid_characters = pickle.load(file)
                return valid_characters
        # DNE
        return None
    
    # Get Kanji characters
    kanji_characters = GetKanjiCharacters(regenerate_kanji_characters)

    # Validate characters
    failed = []
    valid_characters = []

    for idx, ch in enumerate(kanji_characters):

        # Print progress message
        progress_message = f"\rProcessing {idx+1} / {len(kanji_characters)} - {int((idx+1)*100 / len(kanji_characters))}% | {len(valid_characters)} Valid"
        progress_message += " " * (80 - len(progress_message))
        print(progress_message, end="", flush=True)

        # Get stroke numbers property
        stroke_numbers_total = ch.GetProperty("stroke-numbers-total")
        stroke_groups = ch.GetComponents()

        if stroke_groups is None: # No components
            failed.append((ch, "GetComponentImages() accessor yielded \"None\""))
            continue

        if len(stroke_groups) != num_stroke_groups:
            failed.append((ch, f"invalid number of stroke groups: {len(stroke_groups)}"))
            continue

        if stroke_numbers_total is None: # No components
            failed.append((ch, "stroke-numbers-total property query yielded \"None\""))
            continue

        if not HarchImageIsBinary(ch):
            failed.append((ch, "One or more images in image hierarchy was not binary"))
            continue

        groups_ok = True
        strokes_ok = True
        individual_strokes_total = 0
        remaining_groups = [_ for _ in range(10)]
        for group in stroke_groups:

            group_type = group.GetProperty("type")
            group_index = group.GetProperty("group-index")
            group_types = group.GetProperty("group-types")

            if group_type is None: # No type
                failed.append((ch, "component type property query yielded \"None\""))
                groups_ok = False
                break

            elif group_type != "stroke-group": # Not stroke-group
                failed.append((ch, f"component type \"{group_type}\" not \"stroke-group\""))
                groups_ok = False
                break

            elif group_index is None: # No group index
                failed.append((ch, "group index property query yielded \"None\""))
                groups_ok = False
                break
            
            elif group_types is None: # No group types list
                failed.append((ch, "group type list property query yielded \"None\""))
                groups_ok = False
                break

            if group_index not in range(10):
                failed.append((ch, f"out of range group index {group_index}"))
                groups_ok = False
                break

            remaining_groups.remove(group_index)

            strokes = group.GetComponents()
            if strokes is not None:
                for stroke in strokes:
                    stroke_type = stroke.GetProperty("stroke-type")

                    if stroke_type is None:
                        strokes_ok = False
                        failed.append((ch, "stroke type query yielded \"None\""))
                        break

                    elif stroke_type not in groupings.keys():
                        strokes_ok = False
                        failed.append((ch, f'Invalid stroke type {stroke_type}'))
                        break

                    elif stroke_type not in group_types:
                        strokes_ok = False
                        failed.append((ch, f"out-of-group stroke type {stroke_type}"))
                        break

                    individual_strokes_total += 1

            if strokes_ok == False: # Invalid stroke -> invalid group
                groups_ok = False
                break

        if groups_ok == False:
            continue

        if individual_strokes_total != stroke_numbers_total:
            failed.append((ch, f'Strokes total mismatch: img:{individual_strokes_total} - strnum:{stroke_numbers_total}'))
            continue

        if len(remaining_groups) > 0:
            failed.append((ch, f'no stroke groups at index(es) {remaining_groups}'))
            continue

        valid_characters.append(ch)
    
    # Print report of processing
    print(f"\nTotal: {len(valid_characters)} of {len(kanji_characters)} Kanji character valid")
    print(f'Skipped: {len(failed)} Invalid Kanji characters')

    # Create output directory if it doesn't exist
    os.makedirs(Output_dir, exist_ok=True)
    
    # Save valid characters
    with open(os.path.join(Output_dir, valid_characters_filename), "wb") as file:
        pickle.dump(valid_characters, file)

    # Save processing info report
    with open(os.path.join(Output_dir, processing_info_filename), "w") as file:
        file.write("<--- Processing Information --->\n\n")
        file.write(f"Kanji character files validated: {len(valid_characters)}\n")
        file.write(f"Kanji character files skipped (invalid): {len(failed)}\n")
        file.write(f"Total Number of Files: {len(kanji_characters)}\n\n")

        file.write("Invalid characters:\n") # All invalid characters
        for (fail, reason) in failed:
            file.write("\t")
            basename = fail.GetProperty("filename")
            if basename is None:
                file.write("(None)")
            else:
                file.write(os.path.basename(basename))
            file.write(f" | {reason}\n")

        file.write("\nKanji Characters validated:\n") # All validated Kanji
        for character in kanji_characters:
            
            file.write("\tName: ") # Character name
            basename = character.GetProperty("character-name")
            if basename is None:
                file.write("(None)")
            else:
                file.write(basename)
            
            file.write(" | Filename: ") # Base filename
            filename = character.GetProperty("filename")
            if filename is None:
                file.write("(None)")
            else:
                file.write(os.path.basename(filename))

            file.write("\n")

        file.write("\nAll Characters:\n") # All files (full path)
        for f in kanji_characters:
            file.write("\t")
            basename = f.GetProperty("filename")
            if basename is None:
                file.write("(None)")
            else:
                file.write(basename)

def MakeTensors(regenerate=False, regenerate_valid_characters=False, regenerate_kanji_characters=False) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Generates and returns pytorch tensors for training segmentation GAN.
    All elements of tensors are 0 or 1, normalized from binary images.

    :param regenerate: Regenerates both `torch.Tensor` objects if `True`, attempts to load tensors from file if `False`.
    :param regenerate_valid_characters: Regenerates image list from validated characters for segmentation if `True`, loads from file if `False`.
    :param regenerate_kanji_characters: Regenerates all images used in tensor generation if `True` and loads from file if `False`. Completely clean regen.
    :returns: Two `torch.Tensor` objects which contains normalized binary images of whole characters and tensor-fields of images of stroke groups.
    """

    # Filenames for character and stroke group tensors
    character_tensors_filename = "segmenter_character_tensors.pt"
    stroke_group_tensors_filename = "segmenter_stroke_group_tensors.pt"

    # Load from file if regen parameter is false
    if regenerate == False and regenerate_valid_characters == False and regenerate_kanji_characters == False:

        # Load from file
        if os.path.exists(os.path.join(Output_dir, character_tensors_filename)): # Character tensors file exists
            character_tensors = torch.load(os.path.join(Output_dir, character_tensors_filename))

            if os.path.exists(os.path.join(Output_dir, stroke_group_tensors_filename)): # Stroke group tensors exist
                stroke_group_tensors = torch.load(os.path.join(Output_dir, stroke_group_tensors_filename))
                
                return character_tensors, stroke_group_tensors # Return tensors
        # DNE
        return None
    
    # Get validated character from KanjiVG
    characters = GetValidCharacters(regenerate=regenerate_valid_characters, regenerate_kanji_characters=regenerate_kanji_characters)

    # Lists to become pytorch tensors
    character_tensors = []
    stroke_group_tensors = []

    # Define transform: PIL.Image.Image -> torch.tensor
    to_tensor = transforms.ToTensor()

    # Process each hierarchal image (character)
    for img in characters:

        character_image = img.GetBaseImage() # Get the base image (whole character image)
        if character_image is None: # Error in character
            return None
        character_tensors.append(to_tensor(character_image)) # Add to list

        component_tensors = [None] * num_stroke_groups # Creates list for all stroke groups
        components = img.GetComponents() # Get all components
        if components is None: # No components
            return None
        
        # Process each component (stroke group)
        for component in components:

            group_image = component.GetBaseImage() # Get component base image (stroke group image)
            if group_image is None: # Error in stroke group
                return None
            
            group_index = component.GetProperty('group-index') # Get index of srtoke group (relative to grouping)
            if group_index is None: # Error in stroke group (contains no index)
                return None
            
            component_tensors[group_index] = to_tensor(group_image).squeeze(0) # Add to list

        stroke_group_tensors.append(torch.stack(component_tensors)) # Stack to full tensor and add to list
    
    # Stack to convert to proper tensor format
    character_tensors = torch.stack(character_tensors).float()
    stroke_group_tensors = torch.stack(stroke_group_tensors).float()

    # Normalize white pixels (255) to 1
    character_tensors[character_tensors > 0] = 1
    stroke_group_tensors[stroke_group_tensors > 0] = 1

    # Create output directory if it doesn't exist
    os.makedirs(Output_dir, exist_ok=True)

    # Save tensors
    torch.save(character_tensors, os.path.join(Output_dir, character_tensors_filename))
    torch.save(stroke_group_tensors, os.path.join(Output_dir, stroke_group_tensors_filename))

    return character_tensors, stroke_group_tensors # Return