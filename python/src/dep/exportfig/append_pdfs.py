from PyPDF2 import PdfMerger
import os

def append_pdfs(output, *args):
    """
    Append multiple PDF files to an existing PDF file.
    
    Parameters:
    output (str): The output PDF file path.
    *args (str): The PDF files to append.
    """
    # Ensure the output file is writable
    if os.path.exists(output):
        os.remove(output)
    
    merger = PdfMerger()
    
    # Append each PDF
    for pdf in args:
        merger.append(pdf)
    
    merger.write(output)
    merger.close()

# Example usage:
# append_pdfs('output.pdf', 'file1.pdf', 'file2.pdf', 'file3.pdf')
