import subprocess

def pdf2eps(source, dest):
    # Construct the options string for pdftops
    options = ['-q', '-paper', 'match', '-eps', '-level2', source, dest]
    # Convert to eps using pdftops
    try:
        result = subprocess.run(['pdftops'] + options, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if not e.stdout:
            raise RuntimeError('Unable to generate eps. Check destination directory is writable.') from e
        else:
            raise RuntimeError(e.stdout) from e

    # Fix the DSC error created by pdftops
    try:
        with open(dest, 'r+') as fid:
            first_line = fid.readline()
            second_line = fid.readline()
            if second_line.startswith('% Produced by'):
                fid.seek(0, 1)
                fid.write('%')
    except IOError:
        # Cannot open the file
        return

# Example usage:
# pdf2eps('source.pdf', 'dest.eps')
