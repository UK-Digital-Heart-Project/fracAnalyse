import os
import subprocess
import sys

def ghostscript(cmd):
    try:
        # Call ghostscript
        result = subprocess.run(gs_command(gs_path()) + cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        return result.stdout
    except Exception as err:
        # Display possible workarounds for Ghostscript croaks
        url1 = 'https://github.com/altmany/export_fig/issues/12#issuecomment-61467998'  # issue #12
        url2 = 'https://github.com/altmany/export_fig/issues/20#issuecomment-63826270'  # issue #20
        hg2_str = '' 
        if using_hg2():
            hg2_str = ' or Matlab R2014a'
        print(f'Ghostscript error. Rolling back to GS 9.10{hg2_str} may possibly solve this:\n * {url1}')
        if using_hg2():
            print(f'(GS 9.10)\n * {url2} (R2014a)')
        print('\n')

        if sys.platform in ('darwin', 'linux', 'linux2'):
            url3 = 'https://github.com/altmany/export_fig/issues/27'  # issue #27
            print(f'Alternatively, this may possibly be due to a font path issue:\n * {url3}\n')
            # issue #20
            fpath = __file__
            print(f'Alternatively, if you are using csh, modify shell_cmd from "export..." to "setenv ..."\nat the bottom of {fpath}\n')
        raise err

def gs_path():
    # Return a valid path
    path_ = user_string('ghostscript')
    # Check the path works
    if check_gs_path(path_):
        return path_

    # Check whether the binary is on the path
    if os.name == 'nt':
        bin_list = ['gswin32c.exe', 'gswin64c.exe', 'gs']
    else:
        bin_list = ['gs']

    for bin_ in bin_list:
        path_ = bin_
        if check_store_gs_path(path_):
            return path_

    # Search the obvious places
    if os.name == 'nt':
        default_location = 'C:\\Program Files\\gs\\'
        dir_list = os.listdir(default_location)
        if not dir_list:
            default_location = 'C:\\Program Files (x86)\\gs\\'  # Possible location on 64-bit systems
            dir_list = os.listdir(default_location)
        
        executable_list = ['\\bin\\gswin32c.exe', '\\bin\\gswin64c.exe']
        ver_num = 0
        # If there are multiple versions, use the newest
        for dir_name in dir_list:
            try:
                ver_num2 = float(dir_name[2:])
            except ValueError:
                continue

            if ver_num2 > ver_num:
                for exe in executable_list:
                    path2 = os.path.join(default_location, dir_name, exe)
                    if os.path.isfile(path2):
                        path_ = path2
                        ver_num = ver_num2

        if check_store_gs_path(path_):
            return path_
    else:
        executable_list = ['/usr/bin/gs', '/usr/local/bin/gs']
        for exe in executable_list:
            path_ = exe
            if check_store_gs_path(path_):
                return path_

    # Ask the user to enter the path
    while True:
        if sys.platform == 'darwin':  # Is a Mac
            print('Ghostscript not found. Please locate the program.')
        
        base = input('Ghostcript not found. Please locate the program: ')
        if not base:
            # User hit cancel or closed window
            break
        base = os.path.join(base, '')
        bin_dirs = ['', 'bin' + os.sep, 'lib' + os.sep]
        for bin_dir in bin_dirs:
            for bin_ in bin_list:
                path_ = os.path.join(base, bin_dir, bin_)
                if os.path.isfile(path_):
                    if check_store_gs_path(path_):
                        return path_

    if sys.platform == 'darwin':
        raise FileNotFoundError('Ghostscript not found. Have you installed it (http://pages.uoregon.edu/koch)?')
    else:
        raise FileNotFoundError('Ghostscript not found. Have you installed it from www.ghostscript.com?')

def check_store_gs_path(path_):
    # Check the path is valid
    good = check_gs_path(path_)
    if not good:
        return False

    # Update the current default path to the path found
    if not user_string('ghostscript', path_):
        filename = os.path.join(os.path.dirname(user_string.__code__.co_filename), '.ignore', 'ghostscript.txt')
        print(f'Warning: Path to ghostscript installation could not be saved in {filename} (perhaps a permissions issue). You can manually create this file and set its contents to {path_}, to improve performance in future invocations (this warning is safe to ignore).')
        return False
    return True

def check_gs_path(path_):
    if not path_:
        return False

    try:
        result = subprocess.run(gs_command(path_) + '-h', shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def gs_command(path_):
    # Initialize any required system calls before calling ghostscript
    # TODO: in Unix/Mac, find a way to determine whether to use "export" (bash) or "setenv" (csh/tcsh)
    shell_cmd = ''
    if sys.platform in ('darwin', 'linux', 'linux2'):
        shell_cmd = 'export LD_LIBRARY_PATH=""; ' if sys.platform == 'linux' else 'export DYLD_LIBRARY_PATH=""; '
    # Construct the command string
    return f'{shell_cmd}"{path_}" '

def using_hg2():
    # Dummy function to emulate MATLAB's using_hg2 function
    return False

def user_string(key, value=None):
    # Dummy function to emulate MATLAB's user_string function
    # This should store and retrieve user-specific settings
    if value is not None:
        # Store the value
        user_settings[key] = value
        return True
    # Retrieve the value
    return user_settings.get(key, '')

user_settings = {}
