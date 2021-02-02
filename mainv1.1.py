import cv2 as cv
import numpy as np
import csv
import os
import pandas as pd
from matplotlib import pyplot as plt

def convert_time_to_sec(time):
    h = int(time.split(":")[0])
    m = int(time.split(":")[1])
    s_ = time.split(":")[2]
    s = int(s_.split(".")[0])
    ms = float(s_.split(".")[1])/100.0

    return h*60*60+m*60+s+ms

def fancy_feature_extractor(img):
    hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()
    return hist

def train():
    start =  []
    end = []
    labels = []
    hists = [[]]
    mean_hists = []
    with open('segmentation01.txt') as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter="\t")
        for row in csvReader:
            msg = row[0].split(' ')
            labels.append(msg[0])
            start.append(msg[1])
            end.append(msg[3])

    stations = sorted(set(labels),key=labels.index)

    start_seconds = [convert_time_to_sec(s) for s in start]
    end_seconds = [convert_time_to_sec(e) for e in end]

    start_frame = [int(s*25) for s in start_seconds]
    end_frame = [int(e*25) for e in end_seconds]
    # Create a VideoCapture object and read from input file
    cap = cv.VideoCapture('recording.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video  file")

        # Read until video is completed
    visit_segment = 0
    frame_cnt = 0
  
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if end_frame[visit_segment] < frame_cnt:
                if(visit_segment < len(start)-1):
                    visit_segment += 1
                hists.append([])
                
            text =""
            if start_frame[visit_segment] <= frame_cnt and end_frame[visit_segment] >= frame_cnt:
                text = labels[visit_segment]
                index = stations.index(text)
                hists[index].append(fancy_feature_extractor(frame))
                
            frame_cnt += 1
        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    
    # Calculate the means
    
    for i in range(len(stations)):
        mean_hist = np.mean(hists[i],axis=0)
        with open('mean_hist.txt','ab') as featureData:
            np.savetxt(featureData,mean_hist,delimiter = ',')
        mean_hists.append(mean_hist)

    time.sleep(3000)
    # Closes all the frames
    cv.destroyAllWindows()
    return None

def cluster():
    start =  []
    end = []
    label = []
    cmp_results = [[],[],[],[],[],[]]
    mean_hists = []

    stations_frame_cnt = [0,0,0,0,0,0]
    stations_error_cnt = [0,0,0,0,0,0]
    with open('segmentation01.txt') as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter="\t")
        for row in csvReader:
            msg = row[0].split(' ')
            label.append(msg[0])
            start.append(msg[1])
            end.append(msg[3])

    start_seconds = [convert_time_to_sec(s) for s in start]
    end_seconds = [convert_time_to_sec(e) for e in end]

    start_frame = [int(s*25) for s in start_seconds]
    end_frame = [int(e*25) for e in end_seconds]

    for i in range(len(start_frame)):
        gap = start_frame[i] - end_frame[i]

    cap = cv.VideoCapture('recording.mp4')
    mean_hists = np.loadtxt("mean_hist.txt").reshape(6, 512)
    stations = ['Entrance','3D-Lab','Kitchen-1','Caffe-area','Lab','Printer-1']

    for i in range(len(start_frame)):
        gap = end_frame[i] - start_frame[i]
        index = stations.index(label[i])
        stations_frame_cnt[index] = stations_frame_cnt[index] + gap

    visit_segment = 0
    frame_cnt = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv.resize(frame, (500,500), interpolation=cv.INTER_AREA)
        font = cv.FONT_HERSHEY_SIMPLEX

        if ret == True:
            if end_frame[visit_segment] < frame_cnt:
                if(visit_segment < len(start)-1):
                    visit_segment += 1
                
                text =""

            if start_frame[visit_segment] <= frame_cnt and end_frame[visit_segment] >= frame_cnt:
                hist = fancy_feature_extractor(frame)
                cmp_cnt = 0

                for mean_hist in mean_hists:
                    mean_hist = np.float32(mean_hist)
                    cmp_result = cv.compareHist(mean_hist,hist,0)
                    cmp_results[cmp_cnt] = cmp_result
                    cmp_cnt = cmp_cnt + 1
                max_cmp = max(cmp_results)
                index = cmp_results.index(max_cmp)
                text = stations[index]
                if text != label[visit_segment]:
                    index = stations.index(label[visit_segment])
                    stations_error_cnt[index] += 1
                 # Display the resulting frame
                cv.rectangle(frame,(0,0),(200,60),(0,0,0),cv.FILLED)
                cv.putText(frame, text, (0,50), font, 1, (255, 255, 255), 3, cv.LINE_AA)

            cv.imshow('Frame', frame)

                # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
            frame_cnt = frame_cnt + 1
        else:
            break
    cap.release()                
    return None
        
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cluster()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
