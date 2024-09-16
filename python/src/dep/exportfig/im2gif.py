import imageio
import numpy as np
import os

def im2gif(A, *args):
    """
    Convert a multiframe image to an animated GIF file.

    Parameters:
    A (str or np.ndarray): Pathname of the input image or HxWxCxN array of images.
    *args: Additional arguments for options.

    Options:
    -nocrop: Do not crop the borders.
    -nodither: Do not use dithering when converting the image.
    -ncolors: Maximum number of colors the GIF can have.
    -loops: Number of times the animation is to be looped.
    -delay: Time in seconds between frames.
    """
    A, options = parse_args(A, *args)

    if options['crop']:
        A = crop_borders(A)

    # Convert to indexed image
    B = []
    for i in range(A.shape[3]):
        B.append(A[:, :, :, i])

    # Save as a gif
    imageio.mimsave(options['outfile'], B, duration=options['delay'], loop=options['loops'])

def parse_args(A, *args):
    """
    Parse the input arguments.

    Parameters:
    A (str or np.ndarray): Pathname of the input image or HxWxCxN array of images.
    *args: Additional arguments for options.

    Returns:
    A (np.ndarray): Array of input images.
    options (dict): Dictionary of options.
    """
    # Set the defaults
    options = {
        'outfile': '',
        'dither': True,
        'crop': True,
        'ncolors': 256,
        'loops': 65535,
        'delay': 1/15
    }

    # Go through the arguments
    a = 0
    n = len(args)
    while a < n:
        if isinstance(args[a], str) and args[a]:
            if args[a][0] == '-':
                opt = args[a][1:].lower()
                if opt == 'nocrop':
                    options['crop'] = False
                elif opt == 'nodither':
                    options['dither'] = False
                elif opt in options:
                    a += 1
                    options[opt] = float(args[a]) if '.' in args[a] else int(args[a])
                else:
                    raise ValueError(f"Option {args[a]} not recognized")
            else:
                options['outfile'] = args[a]
        a += 1

    if not options['outfile']:
        if isinstance(A, str):
            options['outfile'] = os.path.splitext(A)[0] + '.gif'
        else:
            raise ValueError("No output filename given.")

    if isinstance(A, str):
        # Read in the image
        A = imread_rgb(A)

    return A, options

def imread_rgb(name):
    """
    Read image to uint8 RGB array.

    Parameters:
    name (str): Pathname of the image file.

    Returns:
    A (np.ndarray): Array of the image.
    """
    A = imageio.mimread(name)
    if isinstance(A, list):
        A = np.stack(A, axis=3)
    return np.array(A, dtype=np.uint8)

def crop_borders(A):
    """
    Crop borders of the image.

    Parameters:
    A (np.ndarray): Array of the image.

    Returns:
    A (np.ndarray): Cropped array of the image.
    """
    # Implement cropping logic if needed
    return A

# Example usage:
# im2gif('test.tif', '-delay', '0.5')
