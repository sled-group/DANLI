mkdir -p models
cd models
wget -O depth_model.pth "drive.google.com/u/3/uc?id=1eYaYq4KrwhLT3f6e6xz4R8CCoVyCmJlY&export=download&confirm=yes"
wget -O panoptic_model.pth "drive.google.com/u/3/uc?id=1yr8kMgU0cajCbuM6-7Q6WLMX3M1ibcJH&export=download&confirm=yes"
wget -O state_estimator.pth "drive.google.com/u/3/uc?id=1bXdgf7sPU4UCDmyUPaBfZn2mWK34Xrib&export=download&confirm=yes"

mkdir -p subgoal_predictor
cd subgoal_predictor
wget -O tokenizer.zip "drive.google.com/u/3/uc?id=1wE7USoaWGuQ59mIjDBhoqpSbyoM_4j37&export=download&confirm=yes"
unzip -q tokenizer.zip && rm tokenizer.zip
wget -O args.json "drive.google.com/u/3/uc?id=1Rg-VzXy5xEL0aH54zl5lu11C5ZhI3LH7&export=download&confirm=yes"
wget -O ckpt.pth "drive.google.com/u/3/uc?id=1rovXknAdGZ199O0OVre4YGNcAE4e9mpE&export=download&confirm=yes"
