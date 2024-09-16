import os

def user_string(string_name, new_string=None):
    """
    Get/set a user-specific string in a file.

    Parameters:
    string_name (str): The name of the string required, which sets the filename storing the string: <string_name>.txt
    new_string (str, optional): The new string to be saved in the <string_name>.txt file

    Returns:
    str or bool: The currently saved string if getting, or a boolean indicating whether the save was successful if setting.
    """
    if not isinstance(string_name, str):
        raise ValueError('string_name must be a string.')

    # Create the full filename
    fname = f"{string_name}.txt"
    dname = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ignore')
    file_name = os.path.join(dname, fname)

    if new_string is not None:
        # Set string
        if not isinstance(new_string, str):
            raise ValueError('new_string must be a string.')

        # Make sure the save directory exists
        if not os.path.exists(dname):
            try:
                os.makedirs(dname)
                # Make it hidden
                if os.name == 'nt':  # Windows
                    os.system(f'attrib +h {dname}')
            except Exception as e:
                return False

        # Write the file
        try:
            with open(file_name, 'w') as f:
                f.write(new_string)
            return True
        except Exception as e:
            # file cannot be created/updated - use prefdir if file does not already exist
            if not os.path.exists(file_name):
                file_name = os.path.join(os.path.expanduser('~'), '.matplotlib', fname)
                try:
                    with open(file_name, 'w') as f:
                        f.write(new_string)
                    return True
                except Exception as e:
                    return False
            return False
    else:
        # Get string
        try:
            with open(file_name, 'r') as f:
                return f.readline().strip()
        except Exception as e:
            # file cannot be read, try to read the file in prefdir
            file_name = os.path.join(os.path.expanduser('~'), '.matplotlib', fname)
            try:
                with open(file_name, 'r') as f:
                    return f.readline().strip()
            except Exception as e:
                return ''

# Example usage:
# print(user_string('test'))  # Get string
# print(user_string('test', 'This is a test string.'))  # Set string
