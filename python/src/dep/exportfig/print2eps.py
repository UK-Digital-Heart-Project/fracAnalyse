import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
from matplotlib.backends.backend_ps import FigureCanvasPS


def print2eps(name, fig=None, export_options=None, *args):
    options = ['-loose']
    if args:
        options.extend(args)
    if export_options is None:
        export_options = 0
    if fig is None:
        fig = plt.gcf()

    # Default values
    crop_amounts = [np.nan] * 4  # auto-crop all 4 sides by default

    if isinstance(export_options, dict):
        fontswap = export_options.get('fontswap', True)
        bb_crop = export_options.get('crop', 0)
        crop_amounts = export_options.get('crop_amounts', crop_amounts)
        bb_padding = export_options.get('bb_padding', 0)
        renderer = export_options.get('rendererStr', 'opengl')
        if not renderer.startswith('-'):
            renderer = '-' + renderer
    else:
        export_options = np.atleast_1d(export_options)
        fontswap = export_options[2] if len(export_options) > 2 else True
        bb_crop = export_options[1] if len(export_options) > 1 else 0
        bb_padding = export_options[0] if len(export_options) > 0 else 0
        renderer = '-opengl'

    # Construct the filename
    if not name.lower().endswith('.eps'):
        name += '.eps'

    # Set paper size
    fig.set_paper_size('auto')
    fig.set_orientation('portrait')

    # Find all the used fonts in the figure
    font_handles = fig.findobj(lambda x: hasattr(x, 'get_fontname'))
    fonts = [handle.get_fontname() for handle in font_handles]

    # Map supported font aliases onto the correct name
    fontsl = [font.lower().replace(' ', '') for font in fonts]
    font_map = {
        'times': 'times-roman',
        'timesnewroman': 'times-roman',
        'times-roman': 'times-roman',
        'arial': 'helvetica',
        'helvetica': 'helvetica',
        'newcenturyschoolbook': 'newcenturyschlbk',
        'newcenturyschlbk': 'newcenturyschlbk'
    }
    fontsl = [font_map.get(f, f) for f in fontsl]
    fontslu = list(set(fontsl))

    # Determine the font swap table
    if fontswap:
        matlab_fonts = [
            'Helvetica', 'Times-Roman', 'Palatino', 'Bookman', 'Helvetica-Narrow',
            'Symbol', 'AvantGarde', 'NewCenturySchlbk', 'Courier', 'ZapfChancery',
            'ZapfDingbats'
        ]
        matlab_fontsl = [f.lower().replace(' ', '') for f in matlab_fonts]
        require_swap = [f for f in fontslu if f not in matlab_fontsl]
        unused_fonts = [f for f in matlab_fontsl if f not in fontslu]
        font_swap = []
        for a in range(min(len(require_swap), len(unused_fonts))):
            original_font = require_swap[a]
            new_font = unused_fonts[a]
            for idx, font in enumerate(fontsl):
                if font == original_font:
                    font_swap.append((font_handles[idx], new_font, fonts[idx]))

    # Swap the fonts
    for handle, new_font, _ in font_swap:
        handle.set_fontname(new_font)

    # MATLAB bug fix - black and white text can come out inverted sometimes
    black_text_handles = [h for h in fig.findobj(lambda x: isinstance(x, matplotlib.text.Text) and x.get_color() == (0, 0, 0))]
    white_text_handles = [h for h in fig.findobj(lambda x: isinstance(x, matplotlib.text.Text) and x.get_color() == (1, 1, 1))]
    for handle in black_text_handles:
        handle.set_color((0, 0, 0 + np.finfo(float).eps))
    for handle in white_text_handles:
        handle.set_color((1, 1, 1 - np.finfo(float).eps))

    # MATLAB bug fix - white lines can come out funny sometimes
    white_line_handles = [h for h in fig.findobj(lambda x: isinstance(x, matplotlib.lines.Line2D) and x.get_color() == (1, 1, 1))]
    for handle in white_line_handles:
        handle.set_color((1, 1, 1 - 1e-5))

    # Save the figure
    fig.savefig(name, format='eps', papertype='a4', orientation='portrait', bbox_inches='tight')

    # Post-processing on the EPS file
    with open(name, 'r') as file:
        fstrm = file.read()

    # Reset the font and line colors
    for handle in black_text_handles:
        handle.set_color((0, 0, 0))
    for handle in white_text_handles:
        handle.set_color((1, 1, 1))
    for handle in white_line_handles:
        handle.set_color((1, 1, 1))

    # Reset the font names in the figure
    for handle, _, original_font in font_swap:
        handle.set_fontname(original_font)

    if not fstrm:
        print("Loading EPS file failed, so unable to perform post-processing. This is usually because the figure contains a large number of patch objects. Consider exporting to a bitmap format in this case.")
        return

    # Replace the font names
    for _, new_font, original_font in font_swap:
        fstrm = re.sub(re.escape(new_font), original_font.replace(' ', ''), fstrm)

    # Write out the fixed eps file
    with open(name, 'w') as file:
        file.write(fstrm)
        
def eps_maintain_alpha(fig, fstrm=None, stored_colors=None):
    if fstrm is None:  # in: convert transparency in Matlab figure into unique RGB colors
        stored_colors = []
        prop_names = ['facecolor', 'edgecolor']
        for h_obj in fig.get_children():
            for prop_name in prop_names:
                try:
                    color = h_obj.get_facecolor() if prop_name == 'facecolor' else h_obj.get_edgecolor()
                    if len(color) == 4:  # Truecoloralpha
                        n_colors = len(stored_colors)
                        old_color = np.array(color) * 255
                        new_color = np.array([101, 102 + n_colors // 255, n_colors % 255, 255], dtype=np.uint8)
                        stored_colors.append((h_obj, prop_name, old_color, new_color))
                        if prop_name == 'facecolor':
                            h_obj.set_facecolor(new_color / 255)
                        else:
                            h_obj.set_edgecolor(new_color / 255)
                except AttributeError:
                    pass  # Ignore objects without the property or cannot change it
    else:  # restore transparency in Matlab figure by converting back from the unique RGBs
        was_error = False
        n_colors = len(stored_colors)
        found_flags = [False] * n_colors
        for i, colors_data in enumerate(stored_colors):
            h_obj, prop_name, orig_color, new_color = colors_data
            try:
                # Restore the EPS file's patch color
                color_id = ' '.join(map(str, np.round(new_color[:3] / 255, 3)))
                orig_rgb = ' '.join(map(str, np.round(orig_color[:3] / 255, 3)))
                orig_alpha = str(np.round(orig_color[3] / 255, 3))

                # Find and replace the RGBA values within the EPS text fstrm
                if prop_name == 'facecolor':
                    old_str = f'\n{color_id} RC\nN\n'
                    new_str = f'\n{orig_rgb} RC\n{orig_alpha} .setopacityalpha true\nN\n'
                else:  # 'Edge'
                    old_str = f'\n{color_id} RC\n1 LJ\n'
                    new_str = f'\n{orig_rgb} RC\n{orig_alpha} .setopacityalpha true\n'

                found_flags[i] = old_str in fstrm
                fstrm = fstrm.replace(old_str, new_str)

                # Restore the figure object's original color
                if prop_name == 'facecolor':
                    h_obj.set_facecolor(orig_color / 255)
                else:
                    h_obj.set_edgecolor(orig_color / 255)
            except Exception as err:
                if not was_error:
                    print(f'Error maintaining transparency in EPS file: {err}')
                    was_error = True
    if fstrm is None:
        return stored_colors
    else:
        return stored_colors, fstrm, found_flags


# Usage example:
# fig, ax = plt.subplots()
# ax.plot([0, 1, 2], [0, 1, 4])
# print2eps('output', fig)
