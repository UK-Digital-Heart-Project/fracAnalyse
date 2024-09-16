import os
import re
import numpy as np
import scipy.ndimage
from PIL import Image
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

def export_fig(*args):
    imageData, alpha = None, None
    hadError = False
    displaySuggestedWorkarounds = True

    plt.draw()
    plt.pause(0.05)

    fig = plt.gcf()
    fig, options = parse_args(fig, *args)

    if fig is None:
        return None, None
    elif not plt.fignum_exists(fig.number):
        raise ValueError("No figure found")

    cls = all(isinstance(item, (plt.Axes, plt.Patch)) for item in fig.get_children())
    if cls:
        fig = isolate_axes(fig)
    else:
        if not isinstance(fig, plt.Figure):
            raise ValueError("Handle must be that of a figure, axes, or panel")
        old_mode = fig.get_invert_hardcopy()

    magnify = options['magnify'] * options['aa_factor']
    if is_bitmap(options) and magnify != 1:
        font_units = [text for text in fig.findobj(text=True) if text.get_fontunits() == 'normalized']
        if font_units:
            if not cls:
                fig = copy_fig(fig)
                fig.set_visible(False)
                font_units = [text for text in fig.findobj(text=True) if text.get_fontunits() == 'normalized']
                cls = True
            for text in font_units:
                text.set_fontunits('points')

    try:
        Hlims = [ax for ax in fig.get_axes()]
        if not cls:
            Xlims = [ax.get_xlim() for ax in Hlims]
            Ylims = [ax.get_ylim() for ax in Hlims]
            Zlims = [ax.get_zlim() for ax in Hlims if hasattr(ax, 'get_zlim')]
            Xtick = [ax.get_xticks() for ax in Hlims]
            Ytick = [ax.get_yticks() for ax in Hlims]
            Ztick = [ax.get_zticks() for ax in Hlims if hasattr(ax, 'get_zticks')]
            Xlabel = [ax.get_xticklabels() for ax in Hlims]
            Ylabel = [ax.get_yticklabels() for ax in Hlims]
            Zlabel = [ax.get_zticklabels() for ax in Hlims if hasattr(ax, 'get_zticklabels')]

        for ax in Hlims:
            ax.set_xlim(ax.get_xlim(), auto=False)
            ax.set_ylim(ax.get_ylim(), auto=False)
            if hasattr(ax, 'set_zlim'):
                ax.set_zlim(ax.get_zlim(), auto=False)
            ax.set_xticks(ax.get_xticks(), minor=False)
            ax.set_yticks(ax.get_yticks(), minor=False)
            if hasattr(ax, 'set_zticks'):
                ax.set_zticks(ax.get_zticks(), minor=False)

    except Exception as e:
        pass

    try:
        if is_bitmap(options):
            if abs(options['bb_padding']) > 1:
                displaySuggestedWorkarounds = False
                raise ValueError('For bitmap output (png,jpg,tif,bmp) the padding value (-p) must be between -1<p<1')
            
            if options['transparent'] and (options['png'] or options['alpha']):
                B = print_to_array(fig, magnify, 'k')
                B = downsize(B, options['aa_factor'])

                A = print_to_array(fig, magnify, 'w')
                A = downsize(A, options['aa_factor'])

                alpha = np.round(np.sum(B - A, axis=2)) / (255 * 3) + 1
                A = alpha
                A[A == 0] = 1
                A = B / A[:, :, np.newaxis]
                A = A.astype(np.uint8)

                if options['crop']:
                    alpha, vA, vB = crop_borders(alpha, 0, options['bb_padding'], options['crop_amounts'])
                    if not any(np.isnan(vB)):
                        B = np.zeros((alpha.shape[0], alpha.shape[1], A.shape[2]), dtype=np.uint8)
                        B[vB[0]:vB[1], vB[2]:vB[3], :] = A[vA[0]:vA[1], vA[2]:vA[3], :]
                        A = B
                    else:
                        A = A[vA[0]:vA[1], vA[2]:vA[3], :]

                if options['png']:
                    res = options['magnify'] * plt.gcf().dpi / 25.4e-3
                    plt.imsave(f"{options['name']}.png", A, format='png', dpi=res)

                if is_bitmap(options):
                    A = check_greyscale(A)

                if options['alpha']:
                    imageData = A

                if is_bitmap(options):
                    alph = alpha[:, :, np.newaxis]
                    A = (A * alph + 255 * (1 - alph)).astype(np.uint8)

                if options['im']:
                    imageData = A
            else:
                if options['transparent']:
                    pos = fig.get_size_inches() * fig.dpi
                    fig.patch.set_facecolor('w')
                    A = print_to_array(fig, magnify, 'w')
                    fig.patch.set_facecolor('none')
                else:
                    A = print_to_array(fig, magnify, fig.get_facecolor())

                if options['crop']:
                    A = crop_borders(A, fig.get_facecolor(), options['bb_padding'], options['crop_amounts'])

                A = downsize(A, options['aa_factor'])

                if options['colourspace'] == 2:
                    A = rgb_to_grey(A)
                else:
                    A = check_greyscale(A)

                if options['im']:
                    imageData = A

                if options['alpha']:
                    imageData = A
                    alpha = np.zeros((A.shape[0], A.shape[1]), dtype=np.float32)

            if options['png']:
                res = options['magnify'] * plt.gcf().dpi / 25.4e-3
                plt.imsave(f"{options['name']}.png", A, format='png', dpi=res)

            if options['bmp']:
                plt.imsave(f"{options['name']}.bmp", A, format='bmp')

            if options['jpg']:
                quality = options['quality'] if options['quality'] else 95
                if quality > 100:
                    plt.imsave(f"{options['name']}.jpg", A, format='jpg', quality=100, subsampling=0)
                else:
                    plt.imsave(f"{options['name']}.jpg", A, format='jpg', quality=quality)

            if options['tif']:
                if options['colourspace'] == 1 and A.shape[2] == 3:
                    A = 255 - A
                    K = np.min(A, axis=2)
                    C = (A[:, :, 0] - K) * 255 / np.maximum(255 - K, 1)
                    M = (A[:, :, 1] - K) * 255 / np.maximum(255 - K, 1)
                    Y = (A[:, :, 2] - K) * 255 / np.maximum(255 - K, 1)
                    A = np.stack((C, M, Y, K), axis=2).astype(np.uint8)
                plt.imsave(f"{options['name']}.tif", A, format='tif')

        if is_vector(options):
            renderer = 'opengl' if options['renderer'] == 1 else 'zbuffer' if options['renderer'] == 2 else 'painters'
            tmp_name = f"{options['name']}.eps"
            pdf_name_tmp = f"{options['name']}.pdf"

            if options['pdf']:
                pdf_name = f"{options['name']}.pdf"
            else:
                pdf_name = pdf_name_tmp

            p2e_args = [renderer, f"-r{options['resolution']}"]

            if options['colourspace'] == 1:
                p2e_args.append('-cmyk')

            print_to_eps(tmp_name, fig, options, *p2e_args)

            if options['transparent'] and fig.patch.get_facecolor() != 'none':
                remove_background_from_eps(tmp_name, using_hg2(fig))

            if options['colourspace'] == 1:
                change_rgb_to_cmyk(tmp_name)

            if options['bookmark']:
                fig_name = fig.get_label()
                if not fig_name:
                    print("Warning: Bookmark requested for figure with no name. Bookmark will be empty.")
                add_bookmark(tmp_name, fig_name)

            eps_to_pdf(tmp_name, pdf_name_tmp, options['append'], options['colourspace'] == 2, options['quality'], options['gs_options'])

            try:
                move(pdf_name_tmp, pdf_name)
            except Exception as e:
                pass

            if options['eps']:
                try:
                    eps_name_tmp = pdf_name_tmp.replace('.pdf', '.eps')
                    pdf_to_eps(pdf_name, eps_name_tmp)
                    move(eps_name_tmp, f"{options['name']}.eps")
                except Exception as e:
                    if not options['pdf']:
                        remove(pdf_name)
                    try:
                        remove(eps_name_tmp)
                    except Exception as e:
                        pass

                if not options['pdf']:
                    remove(pdf_name)

        if cls or options['close_fig']:
            plt.close(fig)
        else:
            fig.set_invert_hardcopy(old_mode)

            for a, ax in enumerate(Hlims):
                ax.set_xlim(Xlims[a])
                ax.set_ylim(Ylims[a])
                if hasattr(ax, 'set_zlim'):
                    ax.set_zlim(Zlims[a])
                ax.set_xticks(Xtick[a])
                ax.set_yticks(Ytick[a])
                if hasattr(ax, 'set_zticks'):
                    ax.set_zticks(Ztick[a])
                ax.set_xticklabels(Xlabel[a])
                ax.set_yticklabels(Ylabel[a])
                if hasattr(ax, 'set_zticklabels'):
                    ax.set_zticklabels(Zlabel[a])

            for text in texLabels:
                text.set_fontweight('bold')

            for handle, old_unit in zip(annotationHandles, originalUnits):
                handle.set_units(old_unit)

            fig.set_size_inches(oldFigUnits)

        if options['clipboard']:
            try:
                import tkinter as tk
                from PIL import ImageTk, Image
                root = tk.Tk()
                root.withdraw()
                img = Image.fromarray(imageData)
                img_tk = ImageTk.PhotoImage(img)
                root.clipboard_clear()
                root.clipboard_append(img_tk)
                root.update()
                root.destroy()
            except Exception as e:
                print(f"export_fig -clipboard output failed: {e}")

        if not nargout:
            imageData, alpha = None, None

    except Exception as err:
        if displaySuggestedWorkarounds and str(err) != 'export_fig error':
            if not hadError:
                print("export_fig error. Please ensure:")
            print("  that you are using the latest version of export_fig")
            if os.name == 'posix':
                print("  and that you have Ghostscript installed")
            else:
                print("  and that you have Ghostscript installed")
            print("  and that you do not have multiple versions of export_fig installed by mistake")
            print("  and that you did not make a mistake in the expected input arguments")
            if not str(err).startswith('Units'):
                print("  or try to set groot's Units property back to its default value of 'pixels'")
            print("If the problem persists, then please report a new issue.")
        raise err

    return imageData, alpha

def default_options():
    """ Default options used by export_fig """
    options = {
        'name': 'export_fig_out',
        'crop': True,
        'crop_amounts': [np.nan] * 4,  # auto-crop all 4 image sides
        'transparent': False,
        'renderer': 0,  # 0: default, 1: OpenGL, 2: ZBuffer, 3: Painters
        'pdf': False,
        'eps': False,
        'png': False,
        'tif': False,
        'jpg': False,
        'bmp': False,
        'clipboard': False,
        'colourspace': 0,  # 0: RGB/gray, 1: CMYK, 2: gray
        'append': False,
        'im': False,
        'alpha': False,
        'aa_factor': 0,
        'bb_padding': 0,
        'magnify': None,
        'resolution': None,
        'bookmark': False,
        'closeFig': False,
        'quality': None,
        'update': False,
        'fontswap': True,
        'gs_options': []
    }
    return options

def parse_args(nout, fig, *args):
    """ Parse the input arguments """
    # Set the defaults
    native = False  # Set resolution to native of an image
    options = default_options()
    options['im'] = (nout == 1)  # user requested imageData output
    options['alpha'] = (nout == 2)  # user requested alpha output

    # Go through the other arguments
    skipNext = False
    for i, arg in enumerate(args):
        if skipNext:
            skipNext = False
            continue
        if isinstance(arg, (list, tuple)):
            fig = arg
        elif isinstance(arg, str) and arg:
            if arg[0] == '-':
                opt = arg[1:].lower()
                if opt == 'nocrop':
                    options['crop'] = False
                    options['crop_amounts'] = [0, 0, 0, 0]
                elif opt in ['trans', 'transparent']:
                    options['transparent'] = True
                elif opt == 'opengl':
                    options['renderer'] = 1
                elif opt == 'zbuffer':
                    options['renderer'] = 2
                elif opt == 'painters':
                    options['renderer'] = 3
                elif opt == 'pdf':
                    options['pdf'] = True
                elif opt == 'eps':
                    options['eps'] = True
                elif opt == 'png':
                    options['png'] = True
                elif opt in ['tif', 'tiff']:
                    options['tif'] = True
                elif opt in ['jpg', 'jpeg']:
                    options['jpg'] = True
                elif opt == 'bmp':
                    options['bmp'] = True
                elif opt == 'rgb':
                    options['colourspace'] = 0
                elif opt == 'cmyk':
                    options['colourspace'] = 1
                elif opt in ['gray', 'grey']:
                    options['colourspace'] = 2
                elif opt in ['a1', 'a2', 'a3', 'a4']:
                    options['aa_factor'] = int(opt[1])
                elif opt == 'append':
                    options['append'] = True
                elif opt == 'bookmark':
                    options['bookmark'] = True
                elif opt == 'native':
                    native = True
                elif opt == 'clipboard':
                    options['clipboard'] = True
                    options['im'] = True
                    options['alpha'] = True
                elif opt == 'svg':
                    msg = ('SVG output is not supported by export_fig. Use one of the following alternatives:\n'
                           '  1. saveas(gcf,\'filename.svg\')\n'
                           '  2. plot2svg utility: http://github.com/jschwizer99/plot2svg\n'
                           '  3. export_fig to EPS/PDF, then convert to SVG using generic (non-Matlab) tools\n')
                    raise ValueError(msg)
                elif opt == 'update':
                    # Download the latest version of export_fig into the export_fig folder
                    try:
                        zipFileName = 'https://github.com/altmany/export_fig/archive/master.zip'
                        folderName = os.path.dirname(os.path.abspath(__file__))
                        targetFileName = os.path.join(folderName, f"{np.datetime_as_string(np.datetime64('now'), unit='D')}.zip")
                        urlretrieve(zipFileName, targetFileName)
                    except Exception as e:
                        raise RuntimeError(f"Could not download {zipFileName} into {targetFileName}\n") from e

                    # Unzip the downloaded zip file in the export_fig folder
                    try:
                        from zipfile import ZipFile
                        with ZipFile(targetFileName, 'r') as zip_ref:
                            zip_ref.extractall(folderName)
                    except Exception as e:
                        raise RuntimeError(f"Could not unzip {targetFileName}\n") from e
                elif opt == 'nofontswap':
                    options['fontswap'] = False
                else:
                    try:
                        if opt.startswith('-d'):
                            options['gs_options'].append(f"-d{opt[2:]}")
                        elif opt.startswith('-c'):
                            if len(opt) == 2:
                                skipNext = True
                                vals = list(map(float, args[i + 1]))
                            else:
                                vals = list(map(float, opt[2:].split(',')))
                            if len(vals) != 4:
                                raise ValueError('option -c cannot be parsed: must be a 4-element numeric vector')
                            options['crop_amounts'] = vals
                            options['crop'] = True
                        else:  # scalar parameter value
                            val = float(re.search(r'(?<=-(m|r|q|p))-?\d*\.?\d+', opt).group(0))
                            if np.isnan(val):
                                val = float(args[i + 1])
                                skipNext = True
                            if np.isnan(val):
                                raise ValueError(f"option {arg} is not recognised or cannot be parsed")
                            if opt[1] == 'm':
                                if val <= 0:
                                    raise ValueError(f"Bad magnification value: {val} (must be positive)")
                                options['magnify'] = val
                            elif opt[1] == 'r':
                                options['resolution'] = val
                            elif opt[1] == 'q':
                                options['quality'] = max(val, 0)
                            elif opt[1] == 'p':
                                options['bb_padding'] = val
                    except Exception as err:
                        raise ValueError(f"Unrecognized export_fig input option: '{arg}'") from err
            else:
                options['name'], ext = os.path.splitext(arg)
                if ext:
                    if ext.lower() in ['.tif', '.tiff']:
                        options['tif'] = True
                    elif ext.lower() in ['.jpg', '.jpeg']:
                        options['jpg'] = True
                    elif ext.lower() == '.png':
                        options['png'] = True
                    elif ext.lower() == '.bmp':
                        options['bmp'] = True
                    elif ext.lower() == '.eps':
                        options['eps'] = True
                    elif ext.lower() == '.pdf':
                        options['pdf'] = True
                    elif ext.lower() == '.fig':
                        # If no open figure, then load the specified .fig file and continue
                        if fig is None:
                            fig = openfig(arg, 'invisible')
                            args[i] = fig
                            options['closeFig'] = True
                        else:
                            # save the current figure as the specified .fig file and exit
                            saveas(fig[0], arg)
                            fig = -1
                            return fig, options
                    elif ext.lower() == '.svg':
                        msg = ('SVG output is not supported by export_fig. Use one of the following alternatives:\n'
                               '  1. saveas(gcf,\'filename.svg\')\n'
                               '  2. plot2svg utility: http://github.com/jschwizer99/plot2svg\n'
                               '  3. export_fig to EPS/PDF, then convert to SVG using generic (non-Matlab) tools\n')
                        raise ValueError(msg)
                    else:
                        options['name'] = arg

    # Quick bail-out if no figure found
    if fig is None:
        return fig, options

    # Do border padding with respect to a cropped image
    if options['bb_padding']:
        options['crop'] = True

    # Set default anti-aliasing now we know the renderer
    if options['aa_factor'] == 0:
        try:
            isAA = fig.get(ancestor(fig, 'figure'), 'GraphicsSmoothing') == 'on'
        except:
            isAA = False
        options['aa_factor'] = 1 + 2 * (not isAA or options['renderer'] == 3)

    # Convert user dir '~' to full path
    if len(options['name']) > 2 and options['name'][0] == '~' and options['name'][1] in ['/', '\\']:
        options['name'] = os.path.join(os.path.expanduser('~'), options['name'][1:])

    # Compute the magnification and resolution
    if options['magnify'] is None:
        if options['resolution'] is None:
            options['magnify'] = 1
            options['resolution'] = 864
        else:
            options['magnify'] = options['resolution'] / fig.dpi
    elif options['resolution'] is None:
        options['resolution'] = 864

    # Set the default format
    if not any([options['pdf'], options['eps'], options['png'], options['tif'], options['jpg'], options['bmp']]):
        options['png'] = True

    # Check whether transparent background is wanted (old way)
    if fig.get(ancestor(fig[0], 'figure'), 'Color') == 'none':
        options['transparent'] = True

    # If requested, set the resolution to the native vertical resolution of the first suitable image found
    if native and any([options['png'], options['tif'], options['jpg'], options['bmp']]):
        # Find a suitable image
        list_of_images = fig.find_all(lambda x: x.type == 'image' and x.tag == 'export_fig_native')
        if not list_of_images:
            list_of_images = fig.find_all(lambda x: x.type == 'image' and x.visible)
        for image in list_of_images:
            # Check height is >= 2
            height = image.get_data().shape[0]
            if height < 2:
                continue
            # Account for the image filling only part of the axes, or vice versa
            yl = image.get_ydata()
            if len(yl) == 1:
                yl = [yl[0] - 0.5, yl[0] + height + 0.5]
            else:
                yl = [min(yl), max(yl)]
                if yl[0] == yl[1]:
                    continue
                yl = [yl[0] - 0.5, yl[1] + 0.5 * (yl[1] - yl[0]) / (height - 1)]
            hAx = image.get_parent()
            yl2 = hAx.get_ylim()
            # Find the pixel height of the axes
            oldUnits = hAx.get_units()
            hAx.set_units('pixels')
            pos = hAx.get_position()
            hAx.set_units(oldUnits)
            if pos[3] == 0:
                continue
            # Found a suitable image
            # Account for stretch-to-fill being disabled
            pbar = hAx.get_plotbox_aspect_ratio()
            pos[3] = min(pos[3], pbar[1] * pos[2] / pbar[0])
            # Set the magnification to give native resolution
            options['magnify'] = abs((height * (yl2[1] - yl2[0])) / (pos[3] * (yl[1] - yl[0])))
            break

    return fig, options

def make_cell(A):
    if not isinstance(A, list):
        A = [A]
    return A

def add_bookmark(fname, bookmark_text):
    # Adds a bookmark to the temporary EPS file after %%EndPageSetup
    try:
        with open(fname, 'r') as file:
            fstrm = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f'File {fname} not found.')
    
    # Include standard pdfmark prolog to maximize compatibility
    fstrm = fstrm.replace('%%BeginProlog', '%%%%BeginProlog\n/pdfmark where {pop} {userdict /pdfmark /cleartomark load put} ifelse')
    # Add page bookmark
    fstrm = fstrm.replace('%%EndPageSetup', f'%%%%EndPageSetup\n[ /Title ({bookmark_text}) /OUT pdfmark')

    # Write out the updated file
    try:
        with open(fname, 'w') as file:
            file.write(fstrm)
    except IOError:
        raise IOError(f'Unable to open {fname} for writing.')

def set_tick_mode(Hlims, ax):
    # Set the tick mode of linear axes to manual
    # Leave log axes alone as these are tricky
    M = [getattr(Hlim, f'{ax}Scale') for Hlim in Hlims]
    if not isinstance(M, list):
        M = [M]
    M = [scale == 'linear' for scale in M]
    for i, Hlim in enumerate(Hlims):
        if M[i]:
            setattr(Hlim, f'{ax}TickMode', 'manual')

def change_rgb_to_cmyk(fname):
    # Convert RGB => CMYK within an EPS file
    try:
        fstrm = read_write_entire_textfile(fname)

        # Replace all gray-scale colors
        fstrm = re.sub(r'\n([\d.]+) +GC\n', lambda m: f'\n0 0 0 {1 - float(m.group(1))} CC\n', fstrm)
        
        # Replace all RGB colors
        fstrm = re.sub(r'\n[0.]+ +[0.]+ +[0.]+ +RC\n', '\n0 0 0 1 CC\n', fstrm)  # pure black
        fstrm = re.sub(r'\n([\d.]+) +([\d.]+) +([\d.]+) +RC\n', 
                       lambda m: f'\n{" ".join([str(round(1 - float(x) / max(map(float, m.groups())), 4)) for x in m.groups()])} CC\n', 
                       fstrm)

        read_write_entire_textfile(fname, fstrm)
    except Exception as e:
        # never mind - leave as is...
        pass

def downsize(A, factor):
    # Downsample an image
    if factor == 1:
        # Nothing to do
        return A
    
    try:
        # Faster, but requires Pillow
        A = np.array(Image.fromarray(A).resize((A.shape[1] // factor, A.shape[0] // factor), Image.BILINEAR))
    except ImportError:
        # No Pillow - resize manually
        # Lowpass filter - use Gaussian as is separable, so faster
        # Compute the 1d Gaussian filter
        filt = np.arange(-factor-1, factor+2) / (factor * 0.6)
        filt = np.exp(-filt * filt)
        # Normalize the filter
        filt = filt / np.sum(filt)
        # Filter the image
        padding = len(filt) // 2
        for a in range(A.shape[2]):
            A[:,:,a] = scipy.ndimage.convolve(A[:,:,a], filt[:, None], mode='reflect')
            A[:,:,a] = scipy.ndimage.convolve(A[:,:,a], filt[None, :], mode='reflect')
        # Subsample
        A = A[::factor, ::factor, :]
    
    return A

def rgb2grey(A):
    return (A[:,:,0] * 0.299 + A[:,:,1] * 0.587 + A[:,:,2] * 0.114).astype(A.dtype)

def check_greyscale(A):
    # Check if the image is greyscale
    if A.ndim == 3 and A.shape[2] == 3 and np.all(A[:,:,0] == A[:,:,1]) and np.all(A[:,:,1] == A[:,:,2]):
        A = A[:,:,0] # Save only one channel for 8-bit output
    return A

def eps_remove_background(fname, count):
    # Remove the background of an eps file
    try:
        with open(fname, 'r+') as fh:
            # Read the file line by line
            while count:
                # Get the next line
                l = fh.readline()
                if not l:
                    break # Quit, no rectangle found
                # Check if the line contains the background rectangle
                if re.match(r' *0 +0 +\d+ +\d+ +r[fe] *[\n\r]+', l):
                    # Set the line to whitespace and quit
                    l = re.sub(r'\S', ' ', l)
                    fh.seek(-len(l), 1)
                    fh.write(l)
                    # Reduce the count
                    count -= 1
    except IOError:
        print(f"Not able to open file {fname}.")

def isvector(options):
    return options['pdf'] or options['eps']

def isbitmap(options):
    return any([options['png'], options['tif'], options['jpg'], options['bmp'], options['im'], options['alpha']])

