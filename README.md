# DeepSORT-tracking

# **Project Introduction**

# **Installation**
```
pip install -r requirements.txt
```
Download weights for YOLOv4 and place the file under weights folder. 
The weights can be download from this [link](https://drive.google.com/drive/folders/1IeYXIsRaZBGXuj3PIB-JWbm2nPpNNGqt?usp=sharing)

NOTE: Use .h5 file for tensorflow

# **Run the code**
Use the below command to run the track.py file
```
python track.py --weight weights/yolov4.h5 video --videopath test/test.mp4 --output results/result.mp4
```

The second argument could be changed from 'video' to 'live' with the omission of argument '--videopath' and '--output'.

NOTE: Running YOLO on CPU without the support of GPU can be really slow (~3 fps)
