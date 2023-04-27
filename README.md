# RealGuard
> This is a backend project for face detection and recognition. The frontend project is still under development.
<p>
This project is aimed to provide a real-time face detection and recognition system for security purpose such as door access control. This project is based on <a href="http://dlib.net/">dlib</a> and <a href="http://opencv.org/">OpenCv</a>.
</p>
<p>
You can start the program with a RGBD camera (we use Realsense D435 and D430) or other camera that can generate both 2D and 3D images. Because of the lack of the depth information, 3D images are only used for face validation, which means that the program will judge whether the face is from a real person or a photo. If you don't have a RGBD camera, you can also use a normal camera to start the program, but the face validation will be skipped.
</p>
<p>
Thanks to <a href="http://dlib.net/">dlib</a>, which provides a lot of pretrained models that we can use for our recognition, it saves us much time to train our own models. 
</p>

## Requirements
<p>
Use pip to install the following packages:
</p>

```
pip install opencv-python
pip install dlib
```
<p>
If you want to use Realsense camera, you need to install:
</p>

```
pip install pyrealsense2
```

<p>
Models are also needed, you can download them from <a href="http://dlib.net/files/">here</a>. The models we use are:
</p>

- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [mmod_human_face_detector.dat](http://dlib.net/files/mmod_human_face_detector.dat.bz2)

<p>
Put the models in <code>./model/</code> folder.

## Usage example
<p>
To begin with, put the picture in <code>./face/person_name/</code> folder.
</p>
<p>
Then, run <code>pretrain.py</code> to get the face feature array.
</p>

```
python pretrain.py
```

<p>
If you currently have a RGBD camera, you can start the program with:
</p>

```
python test.py
```

<p>
If you want to work with frontend, you can start with:
</p>

```
python grpcServer.py
```
