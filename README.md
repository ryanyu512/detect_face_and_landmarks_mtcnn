# detect_face_and_landmarks_mtcnn![result](https://user-images.githubusercontent.com/19774686/229136456-6f07b0f3-b60d-429d-ab41-6a420f309495.png)

This project aims at replicate the work of MTCNN: https://kpzhang93.github.io/papers/spl.pdf

1. FDDB (http://vis-www.cs.umass.edu/fddb/) is used for training and validation and testing. 
2. example.ipynb: aims for a quick try of the trained model
3. train_stage.ipynb: aims at training p-net, r-net and o-net
4. hard_sample.ipynb: aims at collecting hard samples for next stage network
5. prepare_FDDB_data.ipynb: aims at convert raw FDDB data into training, validation and testing data
6. marco.py: aims to store less frequently changed global variables  
7. network.py: architechure of the p-net, r-net and o-net
8. mtcnn.py: the pipeline of mtcnn
9. prepare_data.py: custom library for prepare_FDDB_data.ipynb
10. train_model.py: custom library for train_stage.ipynb
11. visualise.py: library for visualisation
