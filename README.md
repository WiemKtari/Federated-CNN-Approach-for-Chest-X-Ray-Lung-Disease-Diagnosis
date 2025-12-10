# Federated-CNN-Approach-for-Chest-X-Ray-Lung-Disease-Diagnosis
This repository contains an experimental study comparing local CNN training and Federated Learning (FL) using the Flower (FLwr) framework for multi-class lung disease classification from chest X-ray images.

The project evaluates how model performance changes when training data is centralized versus distributed across multiple clients, simulating real hospital environments where data privacy is essential.

Federated-CNN-Approach-for-Chest-X-Ray-Lung-Disease-Diagnosis/

│

├── federated/

│   ├── client

|        ├── model.py

|        ├── utils.py

|        ├── client.py

|        ├── requirements.txt

│   ├── server.py

│   ├── model.py

│   ├── requirements.txt

│   └── ...

│

└── local_models/

    ├── client_1.ipynb     # Local model 1
    
    ├── client_2.ipynb     # Local model 2
    
    ├── client_3.ipynb     # Local model 3
    
    └── client_4.ipynb     # Local model 4
    
    
* local_models/ contains CNN models trained separately on each dataset (simulating different hospitals).
* federated/ contains the Flower server and clients used for federated training.

## Datasets Used

Four publicly available chest X-ray datasets were combined to form a diverse multi-source dataset representing different hospitals and imaging conditions.
| Dataset                           | Description                             | Link                                                                                                                                 
| --------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------ |
| **NIH ChestXray14**               | 112,000+ images labeled for 14 diseases | (https://www.kaggle.com/datasets/nih-chest-xrays/data)                                                           
| **COVID-19 Radiography Database** | COVID-19, Normal, Viral Pneumonia       | (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)             
| **Pediatric Pneumonia X-ray**     | Labeled Normal vs Pneumonia (children)  | (https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)                   
| **Tuberculosis X-ray Dataset**    | TB-positive and normal cases            | (https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) 

