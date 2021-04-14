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
    # visit_frames = []
    visit_feature = [] #store features in a visit
    # visit_frame_count_arr = []
    my_file_local_path = "0"+str(index)+"\\segmentation_gt ("+str(index)+").txt" #segmentation_gt() belongs to Max
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, my_file_local_path)
  
    with open(my_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter="\t")
        for row in csvReader:   
            labels.append(row[7])
            start.append(row[1])
            end.append(row[3])

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

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if visit_segment < len(start) and end_frame[visit_segment] < frame_cnt:
                visit_features[visit_segment+arr_idx] = visit_feature
                visit_segment += 1  

                if(visit_segment < len(start)):
                    visit_frame_count.append(0)
                    visit_feature = []
                
            text =""
            if visit_segment < len(start) and start_frame[visit_segment] <= frame_cnt and end_frame[visit_segment] >= frame_cnt:
                if(counter>=5):
                    counter = 0
                    visit_feature.append(extract_features(frame,model))
                    visit_frame_count[visit_segment+arr_idx] += 1 
                    # visit_frames.append(frame) 
            counter += 1               
            frame_cnt += 1
        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    print("feat") #check the length in visit features dict and labels are the same


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
    kmax = 50
    if(kmax>=len(feat)):
        kmax = len(feat)
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2   
    for k in range(2, kmax):
        kmeans = KMeans(n_clusters = k).fit(feat)
        labels = kmeans.labels_
        sil.append(silhouette_score(feat, labels, metric = 'euclidean'))
    return sil.index(max(sil))+2

def PointSelection(DistMatrix,k,n):
    points = []
    for i in range(k):
        minDist = np.inf
        closeIndex = -1
        for j in range(n):
            if DistMatrix[j,0] == i:
                if DistMatrix[j,1] < minDist:
                    minDist = DistMatrix[j,1]
                    closeIndex = j
        points.append(closeIndex)
    return points

def generate_key_frames(feat):
    k = optimal_k(feat)
    kmeans = KMeans(n_clusters = k).fit(feat) # Initialize the clusters 
    
    return kmeans.cluster_centers_

if __name__ == '__main__':
    for index in range(5):
        i = index + 1
        # calculate_visit_features(i)

    # outfile = open("VGG16visit_features",'wb')
    # pickle.dump(visit_features,outfile)
    # outfile.close()

    infile = open("VGG16visit_features",'rb')
    visit_features = pickle.load(infile)
    infile.close()

    visit_segment = 0
    visit_keyframes = {} #Store the key frame
    for visit in visit_features.values():
        #cluster frames in visit to get interia of clusters as keyframe 
        feat = np.array(visit)
        feat = feat.reshape(-1,4096)
        length = len(feat)
        if(length > 2):
            visit_keyframe = generate_key_frames(feat)
            visit_keyframes[visit_segment] = visit_keyframe
        else:
            visit_keyframes[visit_segment] = feat
        visit_segment += 1 

    outfile = open("VGG16visit_keyframes_sil",'wb')
    pickle.dump(visit_keyframes,outfile)
    outfile.close()

    # infile = open("VGG16visit_keyframes_sil",'rb')
    # visit_features = pickle.load(infile)
    # infile.close()
