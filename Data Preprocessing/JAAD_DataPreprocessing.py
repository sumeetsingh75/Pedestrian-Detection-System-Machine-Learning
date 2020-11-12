# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:59:08 2020

@author: sumeet
Description : This program converts the JAAD dataset into PASCAL VOC format to train models present in TenforFlow's Object
Detection API.
To use this file, create a source directory having all raw videos and corresponding XML files.

 ---Source_Directory
|
|-- JAAD_DataPreprocessing.py

This program will read every video file in source direcrtory and creates a temporary directory to save its frames and csv file for annotations. Individual Directories are created as every xml file have same naming conventions for frames.
- Frames with no labelled objects will be discarded.
- Objects labelled as 'people' will be ignored.
- Frames and corresponding Bounding Boxes are scaled down by 0.5.

Finally,data will be merged from individual video directories into a target directory and common csv file will be generated.

"""

#import packages
import glob
import os, sys
import cv2
import pandas as pd
import sys, csv ,operator
import xml.etree.ElementTree as ET
import shutil

#Function parses xml file to extract bounding box info of every frame and store it as a dataframe.
def parse_xml(xml_file):
    #Columns of dataframe.Adding extra column 'key' to store frame number and sort dataframe.
    column_name = ['key','filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    #Extracted info to be appended in a list.
    xml_list = []
    #Reading xml file as a tree structure.
    tree = ET.parse(xml_file)
    #Get root of XML tree.
    root = tree.getroot()
    #Read through all 'track' tags. Every 'track' tag contains BB info of particular object in all frames of video
    for member in root.findall('track'):
        #exclude the info for group of people represented by 'people' label
        if not member.get('label') == 'people':
            #Read through all 'box' tags. Every frame has BB info in 'box' tag.
            for box in member.findall('box'):
                #Read attributes of 'box' tag. Bounding Boxes are scaled down by factor of 0.5
                value = (int(box.get('frame')),("image_"+box.get('frame')+".jpg"),int(root[1][0].find('original_size')[0].text)/2,
                     int(root[1][0].find('original_size')[1].text)/2,("person"),int(float(box.get('xtl'))/2),
                         int(float(box.get('ytl'))/2),int(float(box.get('xbr'))/2),int(float(box.get('ybr'))/2))
                #append values in a list
                xml_list.append(value)
    #convert list to DataFrame
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    #Sort DataFrame using 'key' column to get all BB of one frame together.
    xml_df.sort_values(by=['key'],inplace = True)
    #Dropping 'key' column
    xml_df.drop("key", axis=1, inplace=True)
    return xml_df

# Function to extract frames from video
def generate_frames(src_directory,target_directory,labels):
    global count
    #capture video object
    vidObj = cv2.VideoCapture(src_directory) 
    while True:
        ret,image = vidObj.read() 
        if ret:
            image_name = "image_%d.jpg" % count
            #checking if any labelled object is present in frame to discard non-useful frames.
            if any(labels.filename == image_name):
                #resizing the frame by factor of 0.5.
                image = cv2.resize(image, None, fx=0.50, fy=0.50, interpolation=cv2.INTER_AREA)
                #Saving frames in target directory.
                cv2.imwrite(os.path.join(target_directory ,image_name), image)
        else:
            break;
        count += 1
        
#Function to read data from individual video directories created and merging together into a Target folder and generating common csv file for it.       
def merge_data(src_directory,target_directory):
    #Generate target directiry
    os.mkdir(target_directory)
    #column names for csv file
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    #name of final csv file
    final_csv_name = 'Final_labels.csv'
    #final Dataframe for annotations
    result = pd.DataFrame(columns=column_name)
    name_ptr = 0
    for root,folders,files in os.walk(src_directory):
        #iterating through individual video directories. 
        for subfolder in folders:
            filenumber = 0
            path = os.path.join(src_directory,subfolder)
            #Reading csv file 
            labels = pd.read_csv(os.path.join(path,subfolder[7:] + '_labels.csv'))
            for file in os.listdir(path):
                #Reading image file
                if file.endswith('.jpg'):
                    #selecting every 8th image
                    if filenumber % 8 == 0:
                        #new name of image to make global consistent naming
                        file_newname = "image_%d.jpg" % name_ptr
                        name_ptr = name_ptr + 1
                        #Copying image file with new name to target directory
                        shutil.copyfile(os.path.join(path,file), os.path.join(target_directory,file_newname))
                        #fetching the annatations of the image
                        df = labels[labels['filename'].str.contains(file)]
                        #Renaming file name in annotations DataFrame
                        df["filename"].replace({file:file_newname}, inplace=True)
                        #Concatinf result in final DataFrame
                        result = pd.concat([result, df], axis=0)
                    filenumber = filenumber + 1
    #Writing DataFrame into csv file.                
    result.to_csv((os.path.join(target_directory,final_csv_name)), index=None)
    #Remove temporary target directory
    shutil.rmtree(src_directory)

if __name__ == '__main__':
    #Get the current working directory
    CWD_PATH = os.getcwd()
    #Raw data folder consisting of videos and annotations
    RAW_DATA_FOLDER = 'raw_data'
    #Target folder consisting of images and corresponding csv file
    TARGET_FOLDER = 'Final_Data'
    #Path to source directory
    PATH_TO_RAW_DATA = os.path.join(CWD_PATH,RAW_DATA_FOLDER)
    
    #Read every video file in source directory  
    for video_file in os.listdir(PATH_TO_RAW_DATA):
        count = 0
        #Temp target directory used for internal processing and gets removed.
        temp_target_folder = 'Extracted_Data'
        if video_file.endswith(".mp4"):
            #creating directory for every video file consisting of its frames and annotations in csv format.
            video_file_directory = 'frames_'+video_file[0:-4]
            video_file_directory_path = os.path.join(CWD_PATH,temp_target_folder,video_file_directory)
            if not os.path.exists(video_file_directory_path):
                os.makedirs(video_file_directory_path)
            
            #Annotation file path for current video file
            annotations_file_path = os.path.join(PATH_TO_RAW_DATA,video_file[0:-3]+"xml")
            #parse xml file and save result as a dataframe
            xml_df = parse_xml(annotations_file_path)
            #csv file name of current processed video
            csv_file_name = video_file[0:-4]+'_labels.csv';
            #csv file path
            csv_file_path = os.path.join(temp_target_folder,'frames_' +video_file[0:-4])
            #writing Dataframe into csv file. 
            xml_df.to_csv((os.path.join(csv_file_path,csv_file_name)), index=None)
            
            #convert current video to frames. Method Args: Source_directory, Target_directory, annotations
            generate_frames(os.path.join(PATH_TO_RAW_DATA, video_file),video_file_directory_path,xml_df)
          
    #merging data into a common directory and generating a common CSV file.
    merge_data(temp_target_folder,TARGET_FOLDER)
    print('finished') 