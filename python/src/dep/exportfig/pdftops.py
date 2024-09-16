import subprocess
import os
import sys
from tkinter import filedialog, messagebox, Tk

def pdftops(cmd):
    """Call pdftops with the given command."""
    path_ = xpdf_path()
    result = subprocess.run(f'"{path_}" {cmd}', shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

def xpdf_path():
    """Return a valid path to pdftops."""
    path_ = user_string('pdftops')
    
    if check_xpdf_path(path_):
        return path_

    if os.name == 'nt':
        bin = 'pdftops.exe'
    else:
        bin = 'pdftops'

    if check_store_xpdf_path(bin):
        return bin

    if os.name == 'nt':
        paths = ['C:\\Program Files\\xpdf\\pdftops.exe', 'C:\\Program Files (x86)\\xpdf\\pdftops.exe']
    else:
        paths = ['/usr/bin/pdftops', '/usr/local/bin/pdftops']

    for path_ in paths:
        if check_store_xpdf_path(path_):
            return path_

    errMsg1 = 'Pdftops not found. Please locate the program, or install xpdf-tools from '
    url1 = 'http://foolabs.com/xpdf'
    print(f'{errMsg1}{url1}')

    errMsg2 = 'If you have pdftops installed, perhaps it is shadowed by another issue as described in '
    url2 = 'https://github.com/altmany/export_fig/issues/137'
    print(f'{errMsg2}{url2}')

    state = 0
    while True:
        if state:
            option1 = 'Install pdftops'
        else:
            option1 = 'Issue #137'
        
        root = Tk()
        root.withdraw()  # Hide the main window
        answer = messagebox.askquestion('Pdftops error', f"{errMsg1}\n\n{errMsg2}", icon='warning')

        if answer == 'Install pdftops':
            webbrowser.open(url1)
        elif answer == 'Issue #137':
            webbrowser.open(url2)
            state = 1
        elif answer == 'Locate pdftops':
            base = filedialog.askdirectory(initialdir='/', title=errMsg1)
            if not base:
                break
            
            bin_dirs = ['', 'bin', 'lib']
            for bin_dir in bin_dirs:
                path_ = os.path.join(base, bin_dir, bin)
                if os.path.isfile(path_) and check_store_xpdf_path(path_):
                    return path_
        else:
            break

    raise FileNotFoundError('pdftops executable not found.')

def check_store_xpdf_path(path_):
    """Check if the path is valid and store it if it is."""
    good = check_xpdf_path(path_)
    if good:
        if not user_string('pdftops', path_):
            print(f'Warning: Path to pdftops executable could not be saved. Enter it manually in .ignore/pdftops.txt.')
    return good

def check_xpdf_path(path_):
    """Check if the given path is a valid pdftops executable."""
    result = subprocess.run(f'"{path_}" -h', shell=True, capture_output=True, text=True)
    message = result.stdout

    good = 'PostScript' in message

    if not good and os.path.isfile(path_):
        print(f'Error running {path_}:\n{message}\n')
    
    return good

def user_string(key, value=None):
    """Simulate getting/setting a user string (e.g., from a config file)."""
    config_path = os.path.join(os.path.expanduser('~'), '.ignore', 'pdftops.txt')
    if value is None:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return file.read().strip()
        return ''
    else:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as file:
            file.write(value)
        return True
