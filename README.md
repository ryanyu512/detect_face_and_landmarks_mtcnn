# detect_face_and_landmarks_mtcnn![result](https://user-images.githubusercontent.com/19774686/229136456-6f07b0f3-b60d-429d-ab41-6a420f309495.png)

This project aims at replicating the work of MTCNN: https://kpzhang93.github.io/papers/spl.pdf

1. FDDB (http://vis-www.cs.umass.edu/fddb/) is used for training and validation and testing. 
2. example.ipynb: aims for a quick try of the trained model
3. camera_example.ipynb: aims for a quick try from the webcam images
4. train_stage.ipynb: aims at training p-net, r-net and o-net
5. hard_sample.ipynb: aims at collecting hard samples for next stage network
6. prepare_FDDB_data.ipynb: aims at convert raw FDDB data into training, validation and testing data
7. marco.py: aims at storeing less frequently changed global variables  
8. network.py: architechure of the p-net, r-net and o-net
9. mtcnn.py: the pipeline of mtcnn
10. prepare_data.py: custom library for prepare_FDDB_data.ipynb
11. train_model.py: custom library for train_stage.ipynb
12. visualise.py: library for visualisation
