import os

path = "/home/eegrad/agupta/projects"
outpath = ".gitignore"
ignore_ext = [".pth", ".jpg", ".png"]
ignore_file = [".gitignore"]
ignore_dir_start = ["."]
ignore_file_in = ["events"]
ignore_dir_in = ["checkpoints", "logs", "data", "dataset"]
ignore_size = '5M'

def write_row(row_path):
    with open(outpath, "a+") as outfile:
        outfile.write(row_path+'\n')

for dirpath, dirnames, filenames in os.walk(path):

    if len(filenames):
        dirname = os.path.dirname(filenames[0])
        if len(dirname):
            if dirname[0] in ignore_dir_start:
                write_row(dirpath)
                continue

            for item in ignore_dir_in:
                if item in dirname:
                    write_row(dirpath)
                    continue

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            if os.path.getsize(filepath)/(1024*1024) > ignore_size:
                write_row(filepath)
                continue
            if filename in ignore_file:
                write_row(filepath)
                continue

            f_name, f_ext = os.path.splitext(filename)
            if f_ext in ignore_ext:
                write_row(filepath)
                continue

            for item in ignore_file_in:
                if item in filename:
                    write_row(filepath)
                    continue
