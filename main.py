import cv2 as cv
import numpy as np
import csv
import os
import pandas as pd

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
# Use the 0-th and 1-st channels
channels = [0, 1]

def convert_time_to_sec(time):
    h = int(time.split(":")[0])
    m = int(time.split(":")[1])
    s_ = time.split(":")[2]
    s = int(s_.split(".")[0])
    ms = float(s_.split(".")[1])/100.0

    return h*60*60+m*60+s+ms

def fancy_feature_extractor(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([frame],[0], None, [256], [0,256])
    cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    return hist

def train():
    start =  []
    end = []
    labels = []
    stations = []
    hists = [[],[],[],[],[],[],[]]
    mean_hists = []

    with open('segmentation.txt') as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter="\t")
        for row in csvReader:
            start.append(row[1])
            end.append(row[3])
            labels.append(row[7])

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
                # with open('feature_extractor.txt','ab') as featureData:  
                # np.savetxt(featureData,features[visit_segment],delimiter=',')
                visit_segment += 1
              
            text =""
            if start_frame[visit_segment] <= frame_cnt and end_frame[visit_segment] >= frame_cnt:
                text = labels[visit_segment]
                if(text in stations):
                    index = stations.index(text)
                    hists[index].append(fancy_feature_extractor(frame))
                else:
                    stations.append(text)
                    index = stations.index(text)
                    hists[index].append(fancy_feature_extractor(frame))

            frame_cnt += 1

            # # Display the resulting frame
            # frame = cv2.resize(frame, (500,500), interpolation=cv2.INTER_AREA)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(frame, text, (0, 100), font, 2.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow('Frame', frame)

            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    
    # Calculate the means
    
    for i in range(len(stations)):
        mean_hist = np.mean(hists[i],axis=0)
        with open('mean.txt','ab') as featureData:
            np.savetxt(featureData,mean_hist,delimiter = ',')
        mean_hists.append(mean_hist)

    time.sleep(3000)
    # Closes all the frames
    cv.destroyAllWindows()
    return None

def cluster():
    start =  []
    end = []
    cmp_results = [[],[],[],[],[],[]]
    mean_hists = []

    with open('segmentation.txt') as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter="\t")
        for row in csvReader:
            start.append(row[1])
            end.append(row[3])

    start_seconds = [convert_time_to_sec(s) for s in start]
    end_seconds = [convert_time_to_sec(e) for e in end]

    start_frame = [int(s*25) for s in start_seconds]
    end_frame = [int(e*25) for e in end_seconds]

    cap = cv.VideoCapture('recording.mp4')
    mean_hists = np.loadtxt("mean.txt").reshape(6, 256)
    stations = ['Entrance','3D-Lab','Kitchen 1','Caffe-area','Lab','Printer 1']

    visit_segment = 0
    frame_cnt = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if end_frame[visit_segment] < frame_cnt:
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
                label = cmp_results.index(max_cmp)
                text = stations[label]
                 # Display the resulting frame
                frame = cv.resize(frame, (500,500), interpolation=cv.INTER_AREA)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame, text, (0, 100), font, 2.5, (255, 255, 255), 2, cv.LINE_AA)
                cv.imshow('Frame', frame)

                # Press Q on keyboard to  exit
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
            frame_cnt + 1
        else:
            break
    cap.release()                
    return None
        
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cluster()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
