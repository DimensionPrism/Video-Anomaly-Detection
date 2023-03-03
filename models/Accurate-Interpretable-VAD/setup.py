import sys
import subprocess

def install_detectron2():
    print("installing detectron2...")
    p = subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/detectron2.git', "--quiet"])
    print("detectron2 installed!")

def install_flownet2():
    p = subprocess.run(["sh", "./utils/preprocess_utils/install_flownet2.sh"])

def install_clip():
    p = subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git", "--quiet"])

install_detectron2()
install_flownet2()
install_clip()