# This script is used to classify objects in a video using the YOLOv8 model.
# The script reads a video from a message queue, classifies the objects in the video, and writes the annotated video to a message queue.
# It saves the detected objects in a json file and the annotated video locally.
# For this it uses the ultralytics package to perform object detection and tracking.

# Local imports
from utils.ReturnObject import ReturnJSON
from utils.TranslateObject import translate
from utils.VariableClass import VariableClass
from utils.ColorDetector import FindObjectColors
from utils.ClassificationObject import ClassificationObject
from utils.AnnotateFrame import annotate_frame, annotate_bbox_frame
from utils.ClassificationObjectFunctions import create_classification_object, edit_classification_object


# External imports
import cv2
import time
import json
import torch
import numpy as np
from ultralytics import YOLO
from uugai_python_dynamic_queue.MessageBrokers import RabbitMQ
from uugai_python_kerberos_vault.KerberosVault import KerberosVault



# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()



# Initialize a message broker using the python_queue_reader package
if var.LOGGING:
    print('Initializing RabbitMQ')
rabbitmq = RabbitMQ(
    queue_name = var.QUEUE_NAME, 
    target_queue_name = var.TARGET_QUEUE_NAME, 
    exchange = var.QUEUE_EXCHANGE, 
    host = var.QUEUE_HOST, 
    username = var.QUEUE_USERNAME,
    password = var.QUEUE_PASSWORD)

# Initialize Kerberos Vault
if var.LOGGING:
    print('Initializing Kerberos Vault')
kerberos_vault = KerberosVault(
    storage_uri = var.STORAGE_URI,
    storage_access_key = var.STORAGE_ACCESS_KEY,
    storage_secret_key = var.STORAGE_SECRET_KEY)


while True:
    # Receive message from the queue, and retrieve the media from the Kerberos Vault utilizing the message information.
    if var.LOGGING:
        print('Receiving message from RabbitMQ')
    message = rabbitmq.receive_message()
    if var.LOGGING:
        print('Retrieving media from Kerberos Vault')
    resp = kerberos_vault.retrieve_media(
        message = message, 
        media_type = 'video', 
        media_savepath = var.MEDIA_SAVEPATH)
    

    if var.TIME_VERBOSE:
        start_time = time.time()


    # Perform object classification on the media
    # initialise the yolo model, additionally use the device parameter to specify the device to run the model on.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL = YOLO(var.MODEL_NAME).to(device)
    if var.LOGGING:
        print(f'Using device: {device}')


    # Open video-capture/recording using the video-path. Throw FileNotFoundError if cap is unable to open.
    if var.LOGGING:
        print(f'Opening video file: {var.MEDIA_SAVEPATH}')
    cap = cv2.VideoCapture(var.MEDIA_SAVEPATH)
    if not cap.isOpened():
        FileNotFoundError('Unable to open video file')


    # Initialize the video-writer if the SAVE_VIDEO is set to True.
    if var.SAVE_VIDEO:
        fourcc = cv2.VideoWriter.fourcc(*'avc1')
        video_out = cv2.VideoWriter(
            filename = var.OUTPUT_MEDIA_SAVEPATH, 
            fourcc = fourcc, 
            fps = var.CLASSIFICATION_FPS, 
            frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )


    if var.FIND_DOMINANT_COLORS:
        color_detector = FindObjectColors(
            downsample_factor = 0.5,
            max_clusters = 6,
            )


    # Initialize the classification process. 
    # 2 lists are initialized:
        # Classification objects
        # Additional list for easy access to the ids.
    classification_object_list: list[ClassificationObject] = []
    classification_object_ids: list[int] = []


    # frame_number -> The current frame number. Depending on the frame_skip_factor this can make jumps. 
    # predicted_frames -> The number of frames, that were used for the prediction. This goes up by one each prediction iteration.
    # frame_skip_factor is the factor by which the input video frames are skipped.
    frame_number, predicted_frames = 0, 0
    frame_skip_factor = int(cap.get(cv2.CAP_PROP_FPS) / var.CLASSIFICATION_FPS)
    

    # Loop over the video frames, and perform object classification.
    # The classification process is done until the counter reaches the MAX_NUMBER_OF_PREDICTIONS or the last frame is reached.
    MAX_FRAME_NUMBER = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if var.LOGGING:
        print(f'Classifying frames')
    while (predicted_frames < var.MAX_NUMBER_OF_PREDICTIONS) and (frame_number < MAX_FRAME_NUMBER):


        # Read the frame from the video-capture.
        success, frame = cap.read()
        if not success:
            break


        # Keep the first frame in memory, if the CREATE_BBOX_FRAME is set to True.
        # This is used to draw the tracking results on.
        if var.CREATE_BBOX_FRAME and frame_number == 0:
            bbox_frame = frame.copy()
    

        # Check if the frame_number corresponds to a frame that should be classified.
        if frame_number % frame_skip_factor == 0:

            # Perform object classification on the frame.
            # persist=True -> The tracking results are stored in the model.
            # persist should be kept True, as this provides unique IDs for each detection.
            # More information about the tracking results via https://docs.ultralytics.com/reference/engine/results/
            results = MODEL.track(
                source=frame, 
                persist=True, 
                verbose=False, 
                conf=var.CLASSIFICATION_THRESHOLD, 
                classes=var.ALLOWED_CLASSIFICATIONS)

            # Check if the results are not None,
            # Otherwise, the postprocessing should not be done.
            # Iterate over the detected objects and their masks.
            if results is not None:
                # Loop over boxes and masks.
                # If no masks are found, meaning the model used is not a segmentation model, the mask is set to None.
                for box, mask in zip(results[0].boxes, results[0].masks or [None] * len(results[0].boxes)):

                    # Check if object are detected.
                    # If no object is detected, the box.id will be None.
                    # In this case, the inner-loop is broken. Not calling the object related functions.
                    if box.id is None:
                        break
                    
                    # Extract the object's id, name, confidence, and trajectory.
                    # Also include the mask, if a segmentation model was used. Otherwise, the mask is set to None.
                    # The crop_and_detect function will use trajectory instead if no mask is provided.
                    object_id = int(box.id)
                    object_name = translate(results[0].names[int(box.cls)])
                    object_conf = float(box.conf)
                    object_trajectory = box.xyxy.tolist()[0]
                    object_mask = np.int32(mask.xy[0].tolist()) if mask is not None else None
                    

                    # Depending on the FIND_DOMINANT_COLORS parameter, the dominant colors are found.
                    if var.FIND_DOMINANT_COLORS:
                        main_colors_bgr, main_colors_hls, main_colors_str = color_detector.crop_and_detect(
                            frame=frame,
                            trajectory=object_trajectory,
                            mask_polygon=object_mask)

                    
                    # Check if the id is already in the classification_object_ids list.
                    # If it is, edit the classification object.
                    # Otherwise, create a new classification object.
                    if object_id in classification_object_ids:
                        edit_classification_object(
                            id=object_id,
                            object_name=object_name,
                            object_conf=object_conf,
                            trajectory=object_trajectory,
                            frame_number=frame_number,
                            classification_object_list=classification_object_list,
                            colors_bgr=main_colors_bgr if var.FIND_DOMINANT_COLORS else None,
                            colors_hls=main_colors_hls if var.FIND_DOMINANT_COLORS else None,
                            colors_str=main_colors_str if var.FIND_DOMINANT_COLORS else None)

                    else:
                        classification_object = create_classification_object(
                            id=object_id,
                            first_object_name=object_name,
                            first_object_conf=object_conf,
                            first_trajectory=object_trajectory,
                            first_frame=frame_number,
                            frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            first_colors_bgr=main_colors_bgr if var.FIND_DOMINANT_COLORS else None,
                            first_colors_hls=main_colors_hls if var.FIND_DOMINANT_COLORS else None,
                            first_colors_str=main_colors_str if var.FIND_DOMINANT_COLORS else None)
                        
                        classification_object_ids.append(object_id)
                        classification_object_list.append(classification_object)


            # Depending on the SAVE_VIDEO or PLOT parameter, the frame is annotated.
            # This is done using a custom annotation function.
            if var.SAVE_VIDEO or var.PLOT:
                annotated_frame = annotate_frame(
                    frame=frame,
                    frame_number=frame_number,
                    classification_object_list=classification_object_list,
                    min_distance=var.MIN_DISTANCE,
                    min_detections=var.MIN_DETECTIONS)
                

                # Show the annotated frame if the PLOT parameter is set to True.
                cv2.imshow("YOLOv8 Tracking", annotated_frame) if var.PLOT else None
                cv2.waitKey(1) if var.PLOT else None
                
                # Write the annotated frame to the video-writer if the SAVE_VIDEO parameter is set to True.
                video_out.write(annotated_frame) if var.SAVE_VIDEO else None

            # Increase the frame_number and predicted_frames by one.
            predicted_frames += 1
        frame_number += 1


    # Depending on the CREATE_BBOX_FRAME parameter, the bbox_frame is annotated.
    # This is done using a custom annotation function.
    if var.CREATE_BBOX_FRAME:
        if var.LOGGING:
            print('Annotating bbox frame')
        bbox_frame = annotate_bbox_frame(
            bbox_frame = bbox_frame, 
            classification_object_list = classification_object_list)


    # Depending on the CREATE_RETURN_JSON parameter, the detected objects are saved in a json file.
    # Initialize the ReturnJSON object.
    # This creates a json object with the correct structure.
    if var.CREATE_RETURN_JSON:
        if var.LOGGING:
            print('Creating ReturnJSON object')
        return_json = ReturnJSON()

        # Depending on the user preference, the detected objects are filtered.
        # In this case, the objects are filtered based on the MIN_DETECTIONS parameters.
        filtered_classification_object_list = []
        for classification_object in classification_object_list:
            if classification_object.occurences >= var.MIN_DETECTIONS:
                filtered_classification_object_list.append(classification_object)
                return_json.add_detected_object(classification_object)
    

    # Depending on the SAVE_RETURN_JSON parameter, the return_json object is saved locally.
    return_json.save_returnjson(var.RETURN_JSON_SAVEPATH) if var.SAVE_RETURN_JSON else None


    # Depending on the SAVE_BBOX_FRAME parameter, the bbox_frame is saved locally.
    cv2.imwrite(var.BBOX_FRAME_SAVEPATH, bbox_frame) if var.SAVE_BBOX_FRAME else None

    
    # Depending on the TARGET_QUEUE_NAME parameter, the resulting JSON-object is sent to the target queue.
    # This is done by adding the data to the original message.
    if var.TARGET_QUEUE_NAME != "":
        message['operation'] = return_json.return_object['operation']
        message['data'] = return_json.return_object['data']
        return_message = json.dumps(message)
        rabbitmq.send_message(return_message)
    

    # Depending on the TIME_VERBOSE parameter, the time it took to classify the objects is printed.
    print(f'Classification took: {round(time.time() - start_time, 1)} seconds, at {var.CLASSIFICATION_FPS} fps.') if var.TIME_VERBOSE else None


    # If the videowriter was active, the videowriter is released. 
    # Close the video-capture and destroy all windows.
    if var.LOGGING:
        print('Releasing video writer and closing video capture')
    video_out.release() if var.SAVE_VIDEO else None
    cap.release()
    cv2.destroyAllWindows()
