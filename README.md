***************************************
Environment
***************************************
Distributor ID:	Ubuntu 
Description:	Ubuntu 14.04.5 LTS 
Release:	14.04 
Codename:	trusty 

Python 2.7.6 (default, Oct 26 2016, 20:30:19) 
[GCC 4.8.4] on linux2 

>>> cv2.__version__
'3.2.0'

cmake version 2.8.12.2


***************************************
Release Notes
***************************************




***************************************
Algorithm
***************************************
It takes a video as INPUT
It renders a video frame as OUTPUT in real-time, ater STRIPE detection with a location with precision of 20 pixel width in 240 pixel width frame. Every positive detection, draws red circles, and a rectangle with text "Zebra" that shows the location with arbitrary accuracy and the red blobs for the intuition on what that positive detection was based upon
It SAVES the frame in USER specified directory as .avi file

for each frame :
-image thresholding -blob detection -stripe filtering based on diameter, inter-blob distance, inter-blob horizontal placement, inter-blob bounding circle max-min overlap
-draw rectangle -write zebra -draw blobs 
