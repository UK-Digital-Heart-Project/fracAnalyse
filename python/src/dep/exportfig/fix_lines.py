import re
import warnings

def fix_lines(fstrm, fname2=None):
    def read_write_entire_textfile(filename, content=None):
        if content is None:
            with open(filename, 'r') as file:
                return file.read()
        else:
            with open(filename, 'w') as file:
                file.write(content)
    
    def using_hg2():
        import matlab.engine
        eng = matlab.engine.start_matlab()
        version = eng.version()
        eng.quit()
        return version >= '8.4.0'  # R2014b is version 8.4.0

    # Issue #20: warn users if using this function in HG2 (R2014b+)
    if using_hg2():
        warnings.warn('The fix_lines function should not be used in this Matlab version.', UserWarning)

    if fname2 is not None or fname2 is not None:
        if fname2 is None:
            # Overwrite the input file
            fname2 = fstrm
        # Read in the file
        fstrm = read_write_entire_textfile(fstrm)

    # Move any embedded fonts after the postscript header
    if fstrm[:15] == '%!PS-AdobeFont-':
        # Find the start and end of the header
        ind = [m.start() for m in re.finditer(r'[\n\r]%!PS-Adobe-', fstrm)]
        ind2 = [m.start() for m in re.finditer(r'[\n\r]%%EndComments[\n\r]+', fstrm)]
        # Put the header first
        if ind and ind2 and ind[0] < ind2[0]:
            fstrm = fstrm[ind[0]+1:ind2[0]] + fstrm[:ind[0]] + fstrm[ind2[0]+1:]

    # Make sure all line width commands come before the line style definitions,
    # so that dash lengths can be based on the correct widths
    # Find all line style sections
    ind = sorted([m.start() for m in re.finditer(r'[\n\r]SO[\n\r]', fstrm)] +
                 [m.start() for m in re.finditer(r'[\n\r]DO[\n\r]', fstrm)] +
                 [m.start() for m in re.finditer(r'[\n\r]DA[\n\r]', fstrm)] +
                 [m.start() for m in re.finditer(r'[\n\r]DD[\n\r]', fstrm)])
    # Find line width commands
    ind2 = [m.start() for m in re.finditer(r'[\n\r]\d* w[\n\r]', fstrm)]
    ind3 = [m.end() for m in re.finditer(r'[\n\r]\d* w[\n\r]', fstrm)]
    # Go through each line style section and swap with any line width commands nearby
    b = 0
    m = len(ind)
    n = len(ind2)
    for a in range(m):
        # Go forwards width commands until we pass the current line style
        while b < n and ind2[b] < ind[a]:
            b += 1
        if b >= n:
            # No more width commands
            break
        # Check we haven't gone past another line style (including SO!)
        if a < m - 1 and ind2[b] > ind[a + 1]:
            continue
        # Are the commands close enough to be confident we can swap them?
        if (ind2[b] - ind[a]) > 8:
            continue
        # Move the line style command below the line width command
        fstrm = fstrm[:ind[a]+1] + fstrm[ind[a]+4:ind3[b]] + fstrm[ind[a]+1:ind[a]+4] + fstrm[ind3[b]:]
        b += 1

    # Find any grid line definitions and change to GR format
    # Find the DO sections again as they may have moved
    ind = [m.start() for m in re.finditer(r'[\n\r]DO[\n\r]', fstrm)]
    if ind:
        # Find all occurrences of what are believed to be axes and grid lines
        ind2 = [m.start() for m in re.finditer(r'[\n\r] *\d* *\d* *mt *\d* *\d* *L[\n\r]', fstrm)]
        if ind2:
            # Now see which DO sections come just before axes and grid lines
            ind2_matrix = [[i2 - i for i in ind] for i2 in ind2]
            ind2_bool = any(any(0 < i < 12 for i in row) for row in ind2_matrix)
            ind = [i for i, b in zip(ind, ind2_bool) if b]
            # Change any regions we believe to be grid lines to GR
            for i in ind:
                fstrm = fstrm[:i+1] + 'GR' + fstrm[i+3:]

    # Define the new styles, including the new GR format
    new_style = [
        '/dom { dpi2point 1 currentlinewidth 0.08 mul add mul mul } bdef',  # Dot length macro based on line width
        '/dam { dpi2point 2 currentlinewidth 0.04 mul add mul mul } bdef',  # Dash length macro based on line width
        '/SO { [] 0 setdash 0 setlinecap } bdef',  # Solid lines
        '/DO { [1 dom 1.2 dom] 0 setdash 0 setlinecap } bdef',  # Dotted lines
        '/DA { [4 dam 1.5 dam] 0 setdash 0 setlinecap } bdef',  # Dashed lines
        '/DD { [1 dom 1.2 dom 4 dam 1.2 dom] 0 setdash 0 setlinecap } bdef',  # Dot dash lines
        '/GR { [0 dpi2point mul 4 dpi2point mul] 0 setdash 1 setlinecap } bdef'  # Grid lines - dot spacing remains constant
    ]

    # Construct the output
    fstrm = re.sub(r'(% line types:.+?)/.+?%', r'\1' + '\n'.join(new_style) + '%', fstrm)

    # Write the output file
    if fname2 is not None or fname2 is not None:
        read_write_entire_textfile(fname2, fstrm)
    else:
        return fstrm

# Usage
# fix_lines('input.ps', 'output.ps')
