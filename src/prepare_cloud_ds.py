import os
import pathlib
from glob import glob

import numpy as np
import rasterio as rio
from skimage import exposure
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def generate_chip_windows(image_height,
                          image_width,
                          chip_size,
                          stride=None,
                          force_image_bounds=True):
    """Generate shapely boxes representing subwindow chips of a rectangular region.

    If the last row/column of chips falls partially beyond the edge of the image
    and force_image_bounds=True, another row/column of boxes is added that ends exactly at the edge.
    In that case, there will be a greater overlap between that extra set of boxes and those from
    the grid.

    Args:
        image_height: int, height of full image
        image_width: int, width of full image
        chip_size: int, chip size in pixels
        stride: int, chip offset. If zero or None, will default to chip_size.
            Note that a stride of zero would yield infinite chips.
            Stride is related to margin as: stride = chip_size - 2 * margin
        force_image_bounds: bool, whether to force the last row and column of chips
            to end at the image edge (thus increases overlap with previous chip)

    Yields:
        shapely boxes
    """
    if image_height < chip_size or image_width < chip_size:
        raise ValueError(
            'Image should be larger than chip size. Got image shape {} and chip size {}'.format(
                (image_height, image_width), chip_size
                ))

    if stride is None or stride == 0:
        stride = chip_size

    if stride > chip_size:
        raise ValueError(
            'Stride should be less than chip size. Got stride: {}, chip size: {}'.format(
                stride, chip_size  
            ))

    chip_start_grid = []
    for ax_size in [image_height, image_width]:
        n_chips, remainder_pixels = np.divmod(ax_size, stride)
        if remainder_pixels > 0:
            n_chips += 1
        start_positions = []
        for chip_number in range(n_chips):
            start = chip_number * stride
            # if the end of the chip goes beyond the image and we
            # are forcing bounds, then:
            if force_image_bounds and start + chip_size > ax_size:
                start_positions.append(ax_size - chip_size)
                # break so that there are no chip duplicates
                break
            else:
                start_positions.append(start)
        # end positions should not go beyond the edge of the image in any case:
        end_positions = [min(s + chip_size, ax_size) for s in start_positions]

        chip_start_grid.append(list(zip(start_positions, end_positions)))

    for xmin, xmax in chip_start_grid[1]:
        for ymin, ymax in chip_start_grid[0]:
            yield xmin, ymin, xmax, ymax


def create_oldclouds_colormap():
    colors = [(0, 0, 0),        # clear (black)
              (204, 204, 255),  # snow (blueberry)
              (64, 64, 64),     # shadow (gray)
              (255, 229, 204),  # haze_light (tan)
              (255, 255, 204),  # haze_heavy (light yellow)
              (255, 255, 255)]  # cloud (white)
    return np.array(colors)


# Taken from earthpy: https://earthpy.readthedocs.io/en/latest/_modules/earthpy/plot.html
# Under BSD3 license: https://github.com/earthlab/earthpy/blob/main/LICENSE
def stretch_im(arr, str_clip):
    """Stretch an image in numpy ndarray format using a specified clip value.

    Parameters
    ----------
    arr: numpy array
        N-dimensional array in rasterio band order (bands, rows, columns)
    str_clip: int
        The % of clip to apply to the stretch. Default = 2 (2 and 98)

    Returns
    ----------
    arr: numpy array with values stretched to the specified clip %

    """
    s_min = str_clip
    s_max = 100 - str_clip
    arr_rescaled = np.zeros_like(arr)
    for ii, band in enumerate(arr):
        lower, upper = np.nanpercentile(band, (s_min, s_max))
        arr_rescaled[ii] = exposure.rescale_intensity(
            band, in_range=(lower, upper)
        )
    return arr_rescaled.copy()


# From earthpy.spatial: https://earthpy.readthedocs.io/en/latest/_modules/earthpy/spatial.html#bytescale
# Under BSD3 license: https://github.com/earthlab/earthpy/blob/main/LICENSE
def bytescale(data, high=255, low=0, cmin=None, cmax=None):
    """Byte scales an array (image).

    Byte scaling converts the input image to uint8 dtype, and rescales
    the data range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    Source code adapted from scipy.misc.bytescale (deprecated in scipy-1.0.0)

    Parameters
    ----------
    data : numpy array
        image data array.
    high : int (default=255)
        Scale max value to `high`.
    low : int (default=0)
        Scale min value to `low`.
    cmin : int (optional)
        Bias scaling of small values. Default is ``data.min()``.
    cmax : int (optional)
        Bias scaling of large values. Default is ``data.max()``.

    Returns
    -------
    img_array : uint8 numpy array
        The byte-scaled array.

    Examples
    --------
        >>> import numpy as np
        >>> from earthpy.spatial import bytescale
        >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
        ...                 [ 73.88003259,  80.91433048,   4.88878881],
        ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
        >>> bytescale(img)
        array([[255,   0, 236],
               [205, 225,   4],
               [140,  90,  70]], dtype=uint8)
        >>> bytescale(img, high=200, low=100)
        array([[200, 100, 192],
               [180, 188, 102],
               [155, 135, 128]], dtype=uint8)
        >>> bytescale(img, cmin=0, cmax=255)
        array([[255,   0, 236],
               [205, 225,   4],
               [140,  90,  70]], dtype=uint8)
    """
    if data.dtype == "uint8":
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None or (cmin < data.min()):
        cmin = float(data.min())

    if (cmax is None) or (cmax > data.max()):
        cmax = float(data.max())

    # Calculate range of values
    crange = cmax - cmin
    if crange < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif crange == 0:
        raise ValueError(
            "`cmax` and `cmin` should not be the same value. Please specify "
            "`cmax` > `cmin`"
        )

    scale = float(high - low) / crange

    # If cmax is less than the data max, then this scale parameter will create
    # data > 1.0. clip the data to cmax first.
    data[data > cmax] = cmax
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype("uint8")


chip_size = 224 * 2 # Handle the fact that cropping will reduce image size by 2.

in_path = "/home/bradneuberg/datasets/cloud_data_combined"
out_path = f"/home/bradneuberg/datasets/cloud_data_combined_{chip_size}x{chip_size}"

paths = ["training", "validation"]

# Whether to generate a PNG with a viewable label preview, with a color per
# class.
generate_previews = True

label_cmap = create_oldclouds_colormap()

# Set to -1 to have no limit, useful for debugging.
image_limit = -1
window_limit = -1

for p in paths:
    full_input_path = os.path.join(in_path, p)
    print(f"\tDealing with {full_input_path}...")
    full_image_output_path = os.path.join(out_path, p, "images")
    full_label_output_path = os.path.join(out_path, p, "labels")
    pathlib.Path(full_image_output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(full_label_output_path).mkdir(parents=True, exist_ok=True)

    images = glob(os.path.join(full_input_path, "images", "*.tif"))
    if image_limit != -1:
        images = images[:image_limit]

    labels = []
    for image_path in images:
        label_path = (image_path.replace("images", "labels")
                                .replace("__X", "__y")
                                .replace(".toar", ""))
        labels.append(label_path)

    for img_path, label_path in tqdm(zip(images, labels)):
        with rio.open(img_path, "r") as img_f, \
             rio.open(label_path, "r") as label_f:
            img_data = img_f.read()
            label_data = label_f.read()

            assert img_data.shape[1] == label_data.shape[1], \
                f"The image data height {img_data.shape[1]} != label data height {label_data.shape[1]}"
            assert img_data.shape[2] == label_data.shape[2], \
                f"The image data width {img_data.shape[2]} != label data width {label_data.shape[2]}"

            # Pad out images that are too small with black.
            if img_data.shape[1] < chip_size or img_data.shape[2] < chip_size:
                height = max(img_data.shape[1], chip_size)
                width = max(img_data.shape[2], chip_size)
                padded_img_data = np.zeros((img_data.shape[0], height, width),
                                           dtype=img_data.dtype)
                padded_label_data = np.zeros((label_data.shape[0], height, width),
                                             dtype=label_data.dtype)
                padded_img_data[:, :img_data.shape[1], :img_data.shape[2]] = img_data
                padded_label_data[:, :label_data.shape[1], :label_data.shape[2]] = label_data
                img_data = padded_img_data
                label_data = padded_label_data

            img_filename = os.path.splitext(os.path.basename(img_path))[0]
            label_filename = os.path.splitext(os.path.basename(label_path))[0]

            windows = generate_chip_windows(img_data.shape[1], img_data.shape[2], chip_size)
            window_count = 0
            for (xmin, ymin, xmax, ymax) in windows:
                window_count += 1
                if window_limit != -1 and window_count > window_limit:
                    break

                img_results = np.array(img_data[:, ymin:ymax, xmin:xmax])
                label_results = np.array(label_data[:, ymin:ymax, xmin:xmax])

                # If the chip is entirely black, skip it.
                if img_results.sum() == 0:
                    continue

                img_profile = img_f.meta.copy()
                img_profile.update({
                    "width": chip_size,
                    "height": chip_size,
                })

                label_profile = label_f.meta.copy()
                label_profile.update({
                    "width": chip_size,
                    "height": chip_size,
                })

                img_chip_filename = os.path.join(full_image_output_path,
                                                 f"{img_filename}_{xmin}_{xmax}_{ymin}_{ymax}.tif")
                with rio.open(img_chip_filename, 'w', **img_profile) as img_dst:
                    img_dst.colorinterp = img_f.colorinterp
                    img_dst.descriptions = img_f.descriptions
                    img_dst.write(img_results)

                label_chip_filename = os.path.join(full_label_output_path,
                                                   f"{label_filename}_{xmin}_{xmax}_{ymin}_{ymax}.tif")
                with rio.open(label_chip_filename, 'w', **label_profile) as label_dst:
                    label_dst.colorinterp = label_f.colorinterp
                    label_dst.descriptions = label_f.descriptions
                    label_dst.write(label_results)

                if generate_previews:
                    # Generate a preview of the main image window chip.
                    preview_chip = img_results[:3] # Drop NIR channel.
                    rgb = (2,1,0) # BGR ordering.
                    preview_chip = preview_chip[rgb, :, :]
                    preview_chip = stretch_im(preview_chip, str_clip=2)
                    preview_chip = bytescale(preview_chip)
                    mode = "RGB"
                    preview_chip = np.moveaxis(preview_chip, 0, -1) # (C,H,W) -> (H,W,C)
                    preview_img = to_pil_image(preview_chip, mode=mode)
                    preview_filename = f"{img_filename}_{xmin}_{xmax}_{ymin}_{ymax}_preview.png"
                    preview_filename = os.path.join(full_image_output_path, preview_filename)
                    preview_img.save(preview_filename)

                    # Our labels are in C, H, W format, with 6 channels:
                    # clear: 0
                    # snow: 1
                    # shadow: 2
                    # haze_light: 3
                    # haze_heavy: 4
                    # cloud: 5
                    # We need to collapse these into a single label channel
                    # with these class numbers.
                    label = np.zeros((chip_size, chip_size), dtype=np.uint8)
                    for c in range(label_results.shape[0]):
                        view = label_results[c]
                        label[view == 255] = c

                    # Now convert this to a "nice" image with an actual color
                    # per class.
                    preview_label = np.zeros((3, chip_size, chip_size), dtype=np.uint8)
                    preview_label = label_cmap[label].astype(np.uint8)
                    preview_label = to_pil_image(preview_label)
                    preview_filename = f"{label_filename}_{xmin}_{xmax}_{ymin}_{ymax}_preview.png"
                    preview_filename = os.path.join(full_label_output_path, preview_filename)
                    preview_label.save(preview_filename)