import os
import sys
import argparse
import glob
import time
import asyncio
import cv2
import numpy as np
from ultralytics import YOLO



import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

 
load_dotenv()

EMAIL_ADDRESS = os.getenv('MYMAIL')
EMAIL_PASSWORD = os.getenv('MYPASS')

last_sent = 0
cool_down_time = 5

img_path = 'snapshot.jpg'

def send_email(subject, body, to_email):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg.set_content(body)
    
    with open(img_path, 'rb') as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='snapshot.jpg')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
        
    print("sent mail!")



receiver_email = "chiragjaink07@gmail.com"
mail_content = "Person is Detected!"
mail_heading = "!!!ALERT!!!"





async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                        required=True)
    parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                        image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                        required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                        default=0.5)
    parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                        otherwise, match source resolution',
                        default=None)
    parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                        action='store_true')

    args = parser.parse_args()


    model_path = args.model
    img_source = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record = args.record

    if (not os.path.exists(model_path)):
        print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
        sys.exit(0)

    model = YOLO(model_path, task='detect')
    labels = model.names

    img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
    vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list:
            source_type = 'image'
        elif ext in vid_ext_list:
            source_type = 'video'
        else:
            print(f'File extension {ext} is not supported.')
            sys.exit(0)
    elif 'usb' in img_source:
        source_type = 'usb'
        usb_idx = int(img_source[3:])
    elif 'picamera' in img_source:
        source_type = 'picamera'
        picam_idx = int(img_source[8:])
    else:
        print(f'Input {img_source} is invalid. Please try again.')
        sys.exit(0)

    resize = False
    if user_res:
        resize = True
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

    if record:
        if source_type not in ['video','usb']:
            print('Recording only works for video and camera sources. Please try again.')
            sys.exit(0)
        if not user_res:
            print('Please specify resolution to record video at.')
            sys.exit(0)
        
        record_name = 'demo1.avi'
        record_fps = 30
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

    if source_type == 'image':
        imgs_list = [img_source]
    elif source_type == 'folder':
        imgs_list = []
        filelist = glob.glob(img_source + '/*')
        for file in filelist:
            _, file_ext = os.path.splitext(file)
            if file_ext in img_ext_list:
                imgs_list.append(file)
    elif source_type == 'video' or source_type == 'usb':

        if source_type == 'video': cap_arg = img_source
        elif source_type == 'usb': cap_arg = usb_idx
        cap = cv2.VideoCapture(cap_arg)

        if user_res:
            ret = cap.set(3, resW)
            ret = cap.set(4, resH)

    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
        cap.start()

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    avg_frame_rate = 0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0


    while True:

        t_start = time.perf_counter()

        if source_type == 'image' or source_type == 'folder': 
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                sys.exit(0)
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count = img_count + 1
        
        elif source_type == 'video': 
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
        
        elif source_type == 'usb': 
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
                break

        elif source_type == 'picamera': 
            frame = cap.capture_array()
            if (frame is None):
                print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
                break


        if resize == True:
            frame = cv2.resize(frame,(resW,resH))

        results = model(frame, verbose=False)

        detections = results[0].boxes

        object_count = 0

        for i in range(len(detections)):

            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # Get bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > 0.5:

                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
                # print(classname)
                global last_sent
                current_time=time.time()
                if(current_time - last_sent > cool_down_time):
                    last_sent = current_time
                    if(classname == "person"):
                        cv2.imwrite(img_path, frame)
                        # send_email(mail_heading, mail_content, receiver_email) 
                        asyncio.create_task(asyncio.to_thread(send_email, mail_heading, mail_content, receiver_email)) 
                        await asyncio.sleep(0.2)
                    
                
                # Basic example: count the number of objects in the image
                object_count = object_count + 1

        # Calculate and draw framerate (if using video, USB, or Picamera source)
        if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
        
        # Display detection results
        cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
        cv2.imshow('YOLO detection results',frame) # Display image
        if record: recorder.write(frame)

        # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
        if source_type == 'image' or source_type == 'folder':
            key = cv2.waitKey()
        elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
            key = cv2.waitKey(10)
        
        if key == ord('q') or key == ord('Q'): # Press 'q' to quit
            break
        elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
            cv2.imwrite('capture.png',frame)
        
        # Calculate FPS for this frame
        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))

        # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
        if len(frame_rate_buffer) >= fps_avg_len:
            temp = frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        # Calculate average FPS for past frames
        avg_frame_rate = np.mean(frame_rate_buffer)


    # Clean up
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type == 'video' or source_type == 'usb':
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    if record: recorder.release()
    cv2.destroyAllWindows()

asyncio.run(main())