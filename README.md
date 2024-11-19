# Pancreatic cancer detection from images

| Nr. | Autor(i) / An | Titlul articolului / proiectului | Aplicație / Domeniu | Tehnologii utilizate |  Metodologie  /  Abordare  | Rezultate | Limitări | Comentarii suplimentare |
| --- | ------------- | ------------------------------- | --------------------- | ---------------------- | ----------------------------------------- | ---------- | --------- | ------------------------ |
| 1   | Arshiya S. Ansari, Abu Sarwar Zamani, Mohammad Sajid Mohammadi, Meenakshi, Mahyudin Ritonga, Syed Sohail Ahmed, Devabalan Pounraj, Karthikeyan Kaliyaperumal / 2022|Detection of Pancreatic Cancer in CT Scan Images Using PSO SVM and Image Processing|Medical|PSO SVM, naïve Bayes, and AdaBoost| Detectarea cancerului pancreatic în imaginile CT folosind algoritmi PSO SVM și procesare de imagini. Pentru a elimina zgomotul din imagini, se aplică un filtru de eliminare Gaussian. Imaginea este împărțită în segmente folosind algoritmul K-means. Segmentarea imaginii ajută la identificarea obiectelor și la stabilirea regiunilor de interes. Caracteristicile relevante din imagini sunt extrase folosind algoritmul PCA. Clasificarea se realizează cu ajutorul algoritmilor PSO SVM, Naïve Bayes și AdaBoost. | PSO SVM - 95% acc Naïve Bayes - 87% acc AdaBoost algorithm - 80% acc |Acuratete scazuta?|
| 2   |Po-Ting Chen, Tinghui Wu, Pochuan Wang, Dawei Chang, Kao-Lang Liu, Ming-Shiang Wu, Holger R. Roth, Po-Chang Lee, Wei-Chih Liao, Weichung Wang / 2022|Pancreatic Cancer Detection on CT Scans with Deep Learning: A Nationwide Population-based Study|medical|PyTorch,MONAI,PyTorch Lightning,Tensorboard|Un flux de lucru complet automatizat a fost utilizat pentru a analiza imaginile CT, fără a necesita adnotări sau procesări manuale. Acesta a inclus preprocesarea imaginilor, utilizarea unei rețele neuronale convoluționale (CNN) pentru segmentarea pancreasului și a tumorilor (dacă erau prezente), și un ansamblu de cinci rețele CNN care au clasificat dacă pancreasul avea cancer pancreatic.|a obținut o sensibilitate de 89,9% și o specificitate de 95,9% în setul de testare intern (109 pacienți, 147 subiecți de control), rezultate similare cu sensibilitatea radiologilor (96,1%; P = 0,11).|Fiind antrenat doar cu cazuri de cancer pancreatic, nu poate distinge dintre cancerul pancreatic sau alte malformatii|                          |
| 3   |Anish Gupta; Apeksha Koul; Yogesh Kumar / 2022|Pancreatic Cancer Detection using Machine and Deep Learning Techniques|Medical|Machine and Deep Learning|Cercetările aplică tehnici de machine learning și deep learning pentru diagnosticarea cancerului pancreatic, folosind analize radiomice, date EHR și scanări CT pentru detectarea precoce și prognoza evoluției bolii. Se folosesc algoritmi precum XGBoost, regresie logistică și rețele neuronale, evaluând modele de clasificare și predicție a riscurilor. Aceste metode pot sprijini deciziile clinice și identificarea biomarkerilor.|94%|           |                          |
| 4   |Kaushik Sekaran, P. Chandana & N. Murali Krishna, Seifedine Kadry / 2020|Deep learning convolutional neural network (CNN) With Gaussian mixture model for predicting pancreatic cancer|Medical|Convolutional Neural Network (CNN)Gaussian Mixture Model (GMM)EM Algorithm (Expectation-Maximization)|Metodologia implică utilizarea unui model Gaussian Mixture (GMM) pentru a clasifica datele din imagini CT ale pancreasului, optimizat prin algoritmul EM pentru ajustarea ponderilor și parametrilor componentelor. Pentru detectarea automată a tumorilor, se aplică un Convolutional Neural Network (CNN) cu multiple straturi ascunse, care învață caracteristicile relevante direct din imagini, fără extracție manuală de caracteristici. Această abordare combină analiza statistică și deep learning pentru diagnosticarea eficientă a cancerului pancreatic.|            |           |                          |
| 5   |Kai Cao, Yingda Xia, Jiawen Yao, Xu Han, Lukas Lambert, Tingting Zhang, Wei Tang, Gang Jin, Hui Jiang, Xu Fang, Isabella Nogues, Xuezhou Li, Wenchao Guo, Yu Wang, Wei Fang, Mingyan Qiu, Yang Hou, Tomas Kovarnik, Michal Vocka, Yimei Lu, Yingli Chen, Xin Chen, Zaiyi Liu, Jian Zhou,Jianping Lu / 2023|Large-scale pancreatic cancer detection via non-contrast CT and deep learning|Medical|PANDA, un model AI cu deep learning|Ei au dezvoltat un model AI care a fost antrenat prin supervised machine learning, folosind imaginile CT cu si fara contrast, au dezvoltat un algoritm de segmentare precisa a tumorilor chiar si in absenta contrastului|PANDA a atins o sensibilitate de 95.5% și o specificitate de 99.8%|Sensibilitate redusă pentru leziuni mici sau atipice, posibile rezultate fals-pozitive, fiind nevoie de o validare umană|                          |


## Descriere
Acest proiect folosește o rețea neuronală convoluțională (CNN) pentru a detecta cancerul pancreatic din imagini CT. Scopul este de a clasifica imaginile în două categorii:
- **Normal**: pancreas fără tumori
- **Pancreatic Tumor**: pancreas cu prezența unei tumori

Proiectul este construit cu **PyTorch** și utilizează un model CNN simplu care poate fi antrenat și testat pe un set de imagini structurat corespunzător.

## Structura proiectului
- **main.py**: Codul pentru definirea, antrenarea și salvarea modelului CNN.
- **testare.py**: Script pentru încărcarea modelului salvat și testarea acestuia pe noi imagini.
- **DATASET**: Directorul care conține setul de date organizat în directoare `train` și `test`, fiecare având subdirectoare pentru clasele `normal` și `pancreatic_tumor`.
- **pancreatic_cancer_model.pth**: Fișierul modelului antrenat (va fi generat după rularea `main.py`).

## Setul de date
Asigură-te că ai imaginile CT structurate astfel:
DATASET/  
├── train/  
│   ├── normal/  
│   │   ├── img1.jpg  
│   │   ├── img2.jpg  
│   │   └── ...  
│   └── pancreatic_tumor/  
│       ├── img1.jpg  
│       ├── img2.jpg  
│       └── ...  
└── test/  
    ├── normal/  
    │   ├── img1.jpg  
    │   ├── img2.jpg  
    │   └── ...   
    └── pancreatic_tumor/  
        ├── img1.jpg  
        ├── img2.jpg  
        └── ...     
## Cerințe
Asigură-te că ai instalat următoarele biblioteci:
- Python 3.x
- PyTorch
- Torchvision
- PIL (Python Imaging Library, de obicei inclusă în `Pillow`)

Pentru a instala bibliotecile necesare, rulează:
```bash
pip install torch torchvision pillow
```

## Cum să rulezi proiectul

### 1. Antrenarea modelului
Pentru a antrena modelul, rulează `main.py`. Acesta va citi datele din directorul `DATASET/train`, va antrena modelul și va salva modelul antrenat în `pancreatic_cancer_model.pth`.

```bash
python main.py
```

## 2. Testarea modelului pe o imagine nouă

Pentru a testa modelul pe o imagine nouă, asigură-te că `pancreatic_cancer_model.pth` este generat (după antrenare) și rulează `testare.py`. Modifică calea imaginii în funcția `predict_image` din `testare.py` pentru a specifica imaginea pe care vrei să o testezi.

### Exemplu:
```python
predict_image("DATASET/test/normal/1-081.jpg")
```
Apoi rulează:
```bash
python testare.py
```

Exemplu de output:
```
Prediction: Normal, Confidence: 0.85
```

## Structura Codului

### `main.py`

- **`CNNModel`**: Definiția modelului CNN, care conține straturi convoluționale și complet conectate.
- **`train_model`**: Funcția care antrenează modelul pe setul de date și afișează pierderea și acuratețea la fiecare epocă.
- **Salvarea modelului**: Modelul este salvat în `pancreatic_cancer_model.pth` după antrenare.

### `testare.py`

- Încarcă modelul salvat și aplică transformările necesare pentru imaginea de test.
- **`predict_image`**: Funcția care preprocesează imaginea, face o predicție și afișează rezultatul și nivelul de încredere (confidence).

## Metodologie

- **Preprocesare**: Imaginile sunt redimensionate la 128x128, transformate în tensori și normalizate.
- **Antrenare CNN**: Folosește un model CNN cu două straturi convoluționale și două straturi complet conectate. Funcția de activare finală este Sigmoid pentru a permite clasificarea binară.
- **Inferență**: Modelul prelucrează o imagine de test și estimează probabilitatea de a fi normală sau de a avea o tumoare.

## Rezultate și Limitări

### Rezultate

Modelul poate să facă o clasificare binară între imagini normale și cele cu tumori pancreatice. Performanța va depinde de calitatea și cantitatea setului de date de antrenament.

### Limitări

- Acest model simplu CNN poate să nu fie optim pentru detectarea precisă a cancerului pancreatic, mai ales în cazuri complexe.
- Pentru îmbunătățirea performanței, se pot utiliza arhitecturi CNN mai avansate.
- Necesită un set de date echilibrat și etichetat corect pentru a funcționa eficient.

## Bibliografie

1. **Arshiya S. Ansari et al. / 2022** - *Detection of Pancreatic Cancer in CT Scan Images Using PSO SVM and Image Processing*
2. **Po-Ting Chen et al. / 2022** - *Pancreatic Cancer Detection on CT Scans with Deep Learning: A Nationwide Population-based Study*
3. **Anish Gupta et al. / 2022** - *Pancreatic Cancer Detection using Machine and Deep Learning Techniques*
4. **Kaushik Sekaran et al. / 2020** - *Deep learning convolutional neural network (CNN) With Gaussian mixture model for predicting pancreatic cancer*
5. **Kai Cao et al. / 2023** - *Large-scale pancreatic cancer detection via non-contrast CT and deep learning*
6. **[Sursa imagini - Kaggle ](https://www.kaggle.com/datasets/jayaprakashpondy/pancreatic-ct-images?resource=download)**
7. **[Informatii CNN - PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)**
8. **[Simple Convolutional Neural Network (CNN) for Dummies in PyTorch: A Step-by-Step Guide](https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80)**
## Concluzie

Acest proiect demonstrează utilizarea unui CNN simplu pentru detecția cancerului pancreatic în imagini CT. Pentru rezultate mai bune în aplicații clinice, se recomandă folosirea unor modele avansate și antrenarea pe seturi de date mari și diversificate.
