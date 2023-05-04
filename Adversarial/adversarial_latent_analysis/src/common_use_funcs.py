import os

def create_directory(dir, display=False):
    try:
        # Create target Directory
        os.mkdir(dir)
        if display:
            print("Directory " , dir ,  " Created ") 
    except FileExistsError:
        if display:
            print("Directory " , dir ,  " already exists")