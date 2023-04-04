mkdir -p teach-dataset
cd teach-dataset
wget -O tmp.zip "drive.google.com/u/3/uc?id=1VmwcBoj0Xsz4-smnM-EBmIl_02Dz2u8F&export=download&confirm=yes"
echo "Unzipping..."
unzip -q tmp.zip && rm tmp.zip