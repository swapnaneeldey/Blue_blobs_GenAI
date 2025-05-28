import os
import numpy as np
import tensorflow as tf # Keras is included within TensorFlow
import sys

def image_to_float(file_name):
    """input - file name
    returns normalized float array"""
    # --- Configuration ---
    # 1. Set the path to the folder containing your 256x256 blue blob JPEG images
    image_folder = file_name #

    # 2. Define the expected image size
    target_size = (256, 256)
    img_height, img_width = target_size

    # 3. Define image extensions (only JPEG)
    image_extensions = ('.jpeg')
    # --- End Configuration ---

    print(f"Attempting to load JPEG images from: {image_folder} using TensorFlow/Keras")
    print(f"Expecting images of size: {target_size}")


    image_data_list = []
    filenames_processed = []

    # --- Image Loading Loop ---
    try:
        all_files = os.listdir(image_folder)
    except Exception as e:
        print(f"\nError reading directory '{image_folder}': {e}")
        sys.exit(1)

    print(f"Found {len(all_files)} items in the directory. Processing valid JPEGs...")

    for filename in sorted(all_files):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(image_folder, filename)
            if os.path.isfile(file_path):
                try:
                    # 1. Load image using Keras utility
                    #    - target_size ensures resizing if needed (shouldn't be if already 256x256)
                    #    - color_mode='rgb' ensures 3 channels
                    img = tf.keras.utils.load_img(
                        file_path,
                        color_mode='rgb',
                        target_size=target_size,
                        interpolation='nearest' # Or 'bilinear', 'lanczos'. 'nearest' if exactly 256x256
                    )

                    # 2. Convert the loaded image (PIL format) to a NumPy array
                    #    - Keras defaults to float32, can specify dtype
                    img_array = tf.keras.utils.img_to_array(img, dtype='uint8') # Get uint8 initially

                    # 3. Check shape (optional, but good practice)
                    if img_array.shape != (img_height, img_width, 3):
                        print(f"  Warning: Loaded image '{filename}' has unexpected shape {img_array.shape}. Skipping.")
                        continue

                    # 4. Append the array to the list
                    image_data_list.append(img_array)
                    filenames_processed.append(filename)
                    # print(f"  Loaded {filename}, shape: {img_array.shape}")

                except Exception as e:
                    print(f"  Error processing file '{filename}': {e}")
            else: pass # Ignore subdirectories
        else: pass # Ignore non-JPEGs

    # --- Final Array Conversion ---
    if not image_data_list:
        print("\nError: No valid JPEG images of the specified size were found or processed.")
        sys.exit(1)

    try:
        # Convert list to NumPy array -> shape (num_images, 256, 256, 3)
        blue_blob_data_array = np.array(image_data_list)
    except Exception as e:
        print(f"\nError converting list of images to NumPy array: {e}")
        sys.exit(1)

    # --- Output Results ---
    print("\n------------------------------------")
    print("Image loading successful!")
    print(f"Loaded {blue_blob_data_array.shape[0]} images.")
    # print("Files processed in order:", filenames_processed)
    print(f"Final NumPy array shape: {blue_blob_data_array.shape}")
    print(f"Data type of array elements: {blue_blob_data_array.dtype}") # Will be uint8 here
    print("------------------------------------")

    # Note: As before, you might need to convert to float32 and normalize
    # for your CNN model later:
    blue_blob_data_array_float = blue_blob_data_array.astype(np.float32) / 255.0
    return blue_blob_data_array_float