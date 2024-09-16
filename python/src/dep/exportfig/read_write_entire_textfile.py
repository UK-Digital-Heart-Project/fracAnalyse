def read_write_entire_textfile(fname, fstrm=None):
    """
    Read or write an entire text file to/from memory, without leaving the file open if an error occurs.

    Parameters:
    fname (str): Pathname of the text file to be read.
    fstrm (str, optional): String to be written to the file, including carriage returns.

    Returns:
    str: String read from the file. If fstrm input is given, the output is the same as that input.
    """
    modes = ['r', 'w']
    writing = fstrm is not None
    try:
        with open(fname, modes[writing]) as fh:
            if writing:
                fh.write(fstrm)
                return fstrm
            else:
                return fh.read()
    except Exception as e:
        raise IOError(f"Unable to open or process file {fname}: {str(e)}")

# Example usage:
# content = read_write_entire_textfile('test.txt')  # Read mode
# read_write_entire_textfile('test.txt', 'This is a test string.')  # Write mode
