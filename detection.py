# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import tkinter
import time
import threading


import tkinter as tk

from datetime import datetime
root = tk.Tk()
canvas = tk.Canvas(root, width=200, height=200, borderwidth=0, highlightthickness=0, bg="black")
canvas.grid()

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle


# Create a pipeline
pipeline = rs.pipeline()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#  different resolutions of color and depth streams
config = rs.config()

# from tkinter import *

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 50 #100 meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
skip=0

duration=5
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)


        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        thickness=-1

        color_image = np.asanyarray(color_frame.get_data())

        faces_track = face_cascade.detectMultiScale(color_image,1.3,5)
        #cv2.imshow('color image',color_image)
       

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        # def timer_detection():
        #     threading.Timer(5.0, timer_detection).start()
            
#         def sleeping():
#             if(not len(faces_track)==0):
#                 cv2.circle(images,(25,25),12,(0,255,0),thickness)
#                 cv2.waitkey(500)
#                 continue
        
            
        
        cv2.circle(images,(25,25),12,(0,0,255),thickness)

        # start_time = datetime.now()
        # diff = (datetime.now() - start_time).seconds
        
        def run_loop():

             start_time = datetime.now()
             diff = (datetime.now() - start_time).seconds
             
             while(diff<duration):
                
                cv2.circle(images,(25,25),12,(0,255,0),thickness)
                diff = (datetime.now() - start_time).seconds

        #     #cv2.circle(images,(25,25),12,(0,255,0),thickness)
        #     cv2.waitKey(500)
        #     cv2.circle(images,(25,25),12,(0,255,0),thickness)


        def thread_run():
            t=threading.Thread(target=run_loop)
            t.daemon=True
            t.start()

        # start_time = datetime.now()
        # diff = (datetime.now() - start_time).seconds 

        for face_track in faces_track[-1:]:
            x,y,w,h = face_track
            cv2.rectangle(images,(x,y),(x+w,y+h),(0,255,0),5)
            thread_run()
           
            # cv2.circle(images,(25,25),15,(0,255,0),thickness)
            #cv2.waitKey(500)
            # print(start_time)
            # print(diff)

            # while(diff<duration):
                
            #     cv2.circle(images,(25,25),12,(0,255,0),thickness)
            #     #cv2.imshow('Live Video Frame',images)
            #     diff = (datetime.now() - start_time).seconds

        
        cv2.imshow('Live Video Frame', images)
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
