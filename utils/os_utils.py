import os
import shutil

def link_file(from_file, to_file):
    # This creates a hard link for files on Windows without using mklink.
    # For directories, hard links are not supported, so copying is an alternative.
    if os.path.isfile(from_file):
        os.link(from_file, to_file)
    else:
        # If it's a directory, create a copy instead (since symbolic links require elevated permissions).
        shutil.copytree(from_file, to_file)


def move_file(from_file, to_file):
    # Use shutil.move to move files or directories.
    shutil.move(from_file, to_file)


def copy_file(from_file, to_file):
    # shutil.copy2 will copy a file and its metadata.
    # shutil.copytree will copy an entire directory.
    if os.path.isdir(from_file):
        shutil.copytree(from_file, to_file)
    else:
        shutil.copy2(from_file, to_file)


def remove_file(*fns):
    for f in fns:
        # shutil.rmtree is used to remove directories.
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            # os.remove is used to remove files.
            print(f'Attempting to remove {f}, not found')