#This file can generate visit features and keyframes
# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import pylab as pl
# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import csv
import math

import cv2 as cv
from chord import Chord
from matplotlib import pyplot as plt
from PIL import ImageTk, Image
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, permutations
labels = []
visit_features = {}
visit_frame_count = [] #The amount of a visit frames which is calculated

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

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

def calculate_visit_features(index):
    start =  []
    end = []    
    visit_frame = []
    visit_feature = [] #store features in a visit
    # visit_frame_count_arr = []
    my_file_local_path = "0"+str(index)+"\\segmentation"+str(index)+".txt" #segmentation_gt() belongs to Max
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, my_file_local_path)
  
    with open(my_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter="\t")
        for row in csvReader:   
            labels.append(row[8])
            start.append(row[2])
            end.append(row[4])

    stations_single_video = sorted(set(labels),key=labels.index)

    start_seconds = [convert_time_to_sec(s) for s in start]
    end_seconds = [convert_time_to_sec(e) for e in end]

    start_frame = [int(s*25) for s in start_seconds]
    end_frame = [int(e*25) for e in end_seconds]
    # Create a VideoCapture object and read from input file
    my_video_local_path = "0"+str(index)+"\\recording0"+str(index)+".mp4"
    my_video_file = os.path.join(THIS_FOLDER, my_video_local_path)
    cap = cv.VideoCapture(my_video_file)

    
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

        # Read until video is completed
    visit_segment = 0
    frame_cnt = 0
    
    counter = 0 #For extract frames
    visit_frame_count.append(0)
    arr_idx = len(visit_frame_count)-1
    #Reset the counter at the fist frame in visit
    flag = 0 
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if visit_segment < len(start) and end_frame[visit_segment] < frame_cnt:
                flag = 0
                visit_features[visit_segment+arr_idx] = visit_feature
                visit_segment += 1  
                if(visit_segment < len(start)):
                    visit_frame_count.append(0)
                    visit_feature = []
                
            text =""
            if visit_segment < len(start) and start_frame[visit_segment] <= frame_cnt and end_frame[visit_segment] >= frame_cnt:
                if(flag == 0 or counter>=24):
                    flag = 1
                    counter = 0
                    visit_feature.append(extract_features(frame,model))
                    visit_frame_count[visit_segment+arr_idx] += 1 
            counter += 1 
            # resized = cv.resize(frame,(224,224))   
            # cv.imshow('Frame',resized)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break           
            frame_cnt += 1
        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    print("Finish extracting features from recording0",index,".mp4") #check the length in visit features dict and labels are the same


def extract_features(file, model):
    # load the image as a 224x224 array
    img = cv.cvtColor(file, cv.COLOR_BGR2RGB)
    img = cv.resize(img,(224,224))     
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # convert from 'PIL.Image.Image' to numpy array
    # img = np.array(img) 
 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

# function that lets you view a cluster (based on identifier)   
     
def view_cluster(group):
    plt.figure(figsize = (25,25))
    # gets the list of filenames for a cluster
    files = group
    # only allow up to 30 images to be shown at a time

    # if len(files) > 30:
    #     print(f"Clipping cluster size from {len(files)} to 30")
    #     files = files[:29]

    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(30,10,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

def view_plot(feat,visit_frames,label,visit_segment):
    #Plot the frames
    fig = plt.figure('Visit Plot')
    pca = PCA(n_components=3).fit(feat)
    pca_3d = pca.transform(feat)
    plt.subplot(1,1,1)
    ax = fig.add_subplot(111, projection='3d')
    k_value = optimal_k(feat)
    # kmeans = KMeans(n_clusters = k_value,random_state=111)
    kmeans = KMeans(n_clusters = 1,random_state=111)
    kmeans.fit(pca_3d)

    import os
    VISIT_CENTROID_FOLDER = os.path.dirname(os.path.abspath(__file__))
    visit_centroid_file = os.path.join(VISIT_CENTROID_FOLDER, 'visit_centroid02.txt')
    # ax.scatter(pca_3d[:,0], pca_3d[:,1], pca_3d[:,2], c=kmeans.labels_)
    with open(visit_centroid_file,'ab') as featureData:
        np.savetxt(featureData,kmeans.cluster_centers_[0],delimiter=',')
    print(kmeans.cluster_centers_[0])
   
def optimal_k(feat):
    sil = []
    kmax = 5
    if(kmax>=len(feat)):
        kmax = len(feat)
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2   
    for k in range(2, kmax):
        kmeans = KMeans(n_clusters = k).fit(feat)
        labels = kmeans.labels_
        sil.append(silhouette_score(feat, labels, metric = 'euclidean'))
    return sil.index(max(sil))+2

def euclideanDist(A, B):
    return np.sqrt(sum((A - B) ** 2))

def generate_key_frames(feat):
    k = optimal_k(feat) 

    # length = len(feat)
    # if(length < k):
    #     print("The length of visit frames is less than k value,k = ",k)
    #     return feat

    kmeans = KMeans(n_clusters = k).fit(feat) # Initialize the clusters 
    
    min_dists = [math.inf for i in range(k)]
    min_dists_index = [0 for i in range(k)]
    for index,frame in enumerate(feat):
        label = kmeans.labels_[index]
        center = kmeans.cluster_centers_[label]
        dist = euclideanDist(frame,center)
        if(dist<min_dists[label]):
            min_dists[label] = dist 
            min_dists_index[label] = index

    keyframes = []
    for index in min_dists_index:
        frame = feat[index]
        keyframes.append(frame)        
    return keyframes

def calculate_features():
    # To generate the visit representations
    for index in range(5):
        i = index + 1
        calculate_visit_features(i)

    print("visit features is produced")

    outfile = open("VGG16visit_features",'wb')
    pickle.dump(visit_features,outfile)
    outfile.close()

    print("visit features is produced")

if __name__ == '__main__':
    
    #To calculate features from all videos and output a pickle file for future research
    #If there exists a visit_features file, please comment the code in the next row
    calculate_features()

    infile = open("VGG16visit_features",'rb')
    visit_features = pickle.load(infile)
    infile.close()

    visit_segment = 0
    
    #Use a dict to store keyframe
    visit_keyframes = {} 
    for visit in visit_features.values():
        #cluster frames in visit to get interia of clusters as keyframe 
        feat = np.array(visit)
        feat = feat.reshape(-1,4096)
        visit_keyframe = generate_key_frames(feat)
        visit_keyframes[visit_segment] = visit_keyframe
        # visit_keyframe = generate_key_frames(feat)
        # visit_keyframes[visit_segment] = visit_keyframe
        visit_segment += 1 

    visit_keyframes_filename = "VGG16visit_keyframes" 
    outfile = open(visit_keyframes_filename,'wb')
    pickle.dump(visit_keyframes,outfile)
    outfile.close()
    print("Filename:["+visit_keyframes_filename+"] is produced")
