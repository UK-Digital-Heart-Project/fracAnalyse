import os
import subprocess
import shutil
import tempfile

def eps2pdf(source, dest, crop=True, append=False, gray=False, quality=None, gs_options=None):
    """
    Convert an eps file to pdf format using ghostscript.

    Parameters:
    source (str): filename of the source eps file to convert.
    dest (str): filename of the destination pdf file.
    crop (bool): whether to crop the borders off the pdf. Default: True.
    append (bool): whether the eps should be appended to the end of the pdf as a new page. Default: False.
    gray (bool): whether the output pdf should be grayscale or not. Default: False.
    quality (int): level of image bitmap quality to output. Larger value gives higher quality. Default: None.
    gs_options (list or str): optional ghostscript options. If multiple options are needed, provide as list of strings.
    """
    # Initialize the options string for ghostscript
    options = ['-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=pdfwrite', '-dPDFSETTINGS=/prepress', f'-sOutputFile={dest}']

    # Set crop option
    if crop:
        options.append('-dEPSCrop')

    # Set the font path
    fp = font_path()
    if fp:
        options.append(f'-sFONTPATH={fp}')

    # Set the grayscale option
    if gray:
        options.extend(['-sColorConversionStrategy=Gray', '-dProcessColorModel=/DeviceGray'])

    # Set the bitmap quality
    if quality is not None:
        options.extend(['-dAutoFilterColorImages=false', '-dAutoFilterGrayImages=false'])
        if quality > 100:
            options.extend(['-dColorImageFilter=/FlateEncode', '-dGrayImageFilter=/FlateEncode'])
            options.append('-c ".setpdfwrite << /ColorImageDownsampleThreshold 10 /GrayImageDownsampleThreshold 10 >> setdistillerparams"')
        else:
            options.extend(['-dColorImageFilter=/DCTEncode', '-dGrayImageFilter=/DCTEncode'])
            v = 1 + (quality < 80)
            quality = 1 - quality / 100
            s = f'<< /QFactor {quality:.2f} /Blend 1 /HSample [{v} 1 1 {v}] /VSample [{v} 1 1 {v}] >>'
            options.append(f'-c ".setpdfwrite << /ColorImageDict {s} /GrayImageDict {s} >> setdistillerparams"')

    # Enable users to specify optional ghostscript options
    if gs_options:
        if isinstance(gs_options, list):
            options.extend(gs_options)
        elif isinstance(gs_options, str):
            options.append(gs_options)
        else:
            raise ValueError('gs_options input argument must be a string or list of strings')

    # Check if the output file exists
    if append and os.path.exists(dest):
        # File exists - append current figure to the end
        tmp_nam = tempfile.mktemp()
        try:
            with open(tmp_nam, 'w') as f:
                f.write('1')
        except:
            # Temp dir is not writable, so use the dest folder
            tmp_nam = os.path.join(os.path.dirname(dest), os.path.basename(tmp_nam))

        shutil.copyfile(dest, tmp_nam)
        options.extend(['-f', tmp_nam, source])
        try:
            # Convert to pdf using ghostscript
            result = subprocess.run(['gs'] + options, capture_output=True, text=True)
            status = result.returncode
            message = result.stderr
        finally:
            os.remove(tmp_nam)
    else:
        # File doesn't exist or should be over-written
        options.extend(['-f', source])
        # Convert to pdf using ghostscript
        result = subprocess.run(['gs'] + options, capture_output=True, text=True)
        status = result.returncode
        message = result.stderr

    # Check for error
    if status:
        # Retry without the -sFONTPATH= option
        if fp:
            options = [opt for opt in options if not opt.startswith('-sFONTPATH')]
            result = subprocess.run(['gs'] + options, capture_output=True, text=True)
            if result.returncode == 0:
                return
        raise RuntimeError(f'Ghostscript error: {message}')

def font_path():
    """
    Function to return (and create, where necessary) the font path.
    """
    fp = os.getenv('GS_FONTPATH')
    if not fp:
        if os.name == 'nt':  # Windows
            fp = os.path.join(os.getenv('WINDIR'), 'Fonts')
        else:  # Unix/Linux/Mac
            fp = '/usr/share/fonts:/usr/local/share/fonts:/usr/share/fonts/X11:/usr/local/share/fonts/X11:/usr/share/fonts/truetype:/usr/local/share/fonts/truetype'
    return fp
