import numpy as np
import warnings
import matplotlib.pyplot as plt

def plot_random_images(data_array, num_to_plot=10, min_separation=20, title_prefix="Blue Blob"):
    """
    Plots a specified number of random images from a given data array with a minimum separation constraint.

    Parameters:
        data_array (numpy.ndarray): The array containing image data.
        num_to_plot (int): The number of images to plot.
        min_separation (int): Minimum separation constraint between selected indices.
        title_prefix (str): Prefix for the plot title.

    Returns:
        None
    """
    num_images = data_array.shape[0]
    selected_indices = []
    plot_title_note = ""

    if num_images == 0:
        print(f"No {title_prefix.lower()} images to plot.")
        return
    elif num_images < num_to_plot:
        warnings.warn(f"Requested {num_to_plot} images, but only {num_images} available. Plotting all.")
        selected_indices = list(range(num_images))
        np.random.shuffle(selected_indices)
        num_to_plot = len(selected_indices)
    else:
        available_indices = list(range(num_images))
        np.random.shuffle(available_indices)
        forbidden_ranges = []

        for idx in available_indices:
            is_forbidden = False
            for start, end in forbidden_ranges:
                if start <= idx <= end:
                    is_forbidden = True
                    break

            if not is_forbidden:
                selected_indices.append(idx)
                forbidden_start = max(0, idx - min_separation)
                forbidden_end = min(num_images - 1, idx + min_separation)
                forbidden_ranges.append((forbidden_start, forbidden_end))

                if len(selected_indices) == num_to_plot:
                    plot_title_note = f" (Attempted Separation >= {min_separation})"
                    break

        if len(selected_indices) < num_to_plot:
            warnings.warn(f"Could only find {len(selected_indices)} indices with separation >= {min_separation}. "
                          f"Falling back to simple random sampling for the remaining.")
            plot_title_note = f" (Separation >= {min_separation} failed, used fallback)"

            remaining_needed = num_to_plot - len(selected_indices)
            current_selected_set = set(selected_indices)
            potential_fallback_indices = [i for i in range(num_images) if i not in current_selected_set]
            np.random.shuffle(potential_fallback_indices)
            fallback_indices_to_add = potential_fallback_indices[:remaining_needed]
            selected_indices.extend(fallback_indices_to_add)

            if len(selected_indices) < num_to_plot:
                warnings.warn(f"Could not reach {num_to_plot} total images even with fallback. Plotting {len(selected_indices)}.")

        num_to_plot = len(selected_indices)

    if selected_indices:
        plt.figure(figsize=(12, 5))
        cols = min(5, num_to_plot)
        rows = (num_to_plot + cols - 1) // cols if cols > 0 else 1
        plot_title = f"Random {num_to_plot} {title_prefix} Images" + plot_title_note

        for i, idx in enumerate(selected_indices):
            if i >= rows * cols:
                break
            plt.subplot(rows, cols, i + 1)
            if 0 <= idx < num_images:
                plt.imshow(data_array[idx], cmap='gray')
                plt.title(f"Index: {idx}")
                plt.axis('off')
            else:
                plt.title(f"Invalid Idx: {idx}")
                plt.axis('off')

        plt.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    elif num_images > 0:
        print(f"Failed to select any indices for plotting.")