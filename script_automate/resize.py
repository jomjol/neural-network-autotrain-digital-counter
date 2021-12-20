import glob
import os
from PIL import Image 

def empty_directory(_directory):
    files = glob.glob(_directory + '/*.*')
    i = 0
    for f in files:
        os.remove(f)
        i=i+1
    print(str(i) + " files have been deleted.")
    
    
def image_resize(_from, _to, _target_size_x, _target_size_y):
    files = glob.glob(_from + '/*.bmp')
    files = files + glob.glob(_from + '/*.png')
    files = files + glob.glob(_from + '/*.jpg')
    for aktfile in files:
        test_image = Image.open(aktfile).convert('RGB')
        test_image = test_image.resize((_target_size_x, _target_size_y), Image.NEAREST)
        base=os.path.basename(aktfile)
        base = os.path.splitext(base)[0] + ".jpg"
        save_name = _to + '/' + base
        print("in: " + aktfile + "  -  out: " + save_name)
        test_image.save(save_name, "JPEG")
