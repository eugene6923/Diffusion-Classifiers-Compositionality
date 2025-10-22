import subprocess

def compress_pdf_ghostscript(input_pdf, output_pdf, quality="ebook"):
    gs_command = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/" + quality,
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        "-sOutputFile=" + output_pdf,
        input_pdf
    ]
    subprocess.run(gs_command, check=True)
    print(f"Compressed PDF saved as: {output_pdf}")

# Example Usage
compress_pdf_ghostscript("motivation_others_self_bench3.pdf", "motivation_others_self_bench3_compressed.pdf", quality="screen")

# Example Usage
# compress_pdf("motivation_others_self_bench3.pdf", "motivation_others_self_bench3_compressed.pdf", image_quality=75)
