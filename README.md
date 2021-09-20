<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="App predict/static/images/APP.png" alt="Project logo"></a>
</p>

<h3 align="center">Sentiment Analysis of Cosmetic Product Reviews</h3>



---



## 📝 Table of Contents

- [About](#about)
- [Usage](#usage)
- [Deployment](#deployment)
- [Built Using](#built_using)

## 🧐 About <a name = "about"></a>

Companies can take advantage of sentiment to find out people's feedback on their brand. The Female Daily Review website is one of the platforms used to accommodate all forms of opinion regarding beauty products. The process of retrieving data from the website by implementing web scraping. From the 11119 review data obtained, it is necessary to analyze opinions, emotions, and sentiments by utilizing sentiment analysis that applies text mining to identify and extract a topic. In other words, sentiment analysis can help determine the level of user satisfaction with a cosmetic brand. The algorithm used in this case is 1D-Convolutional Neural Network (1D-CNN). However, before classifying the data, it is necessary to do text preprocessing so that the raw dataset becomes more structured. The results of the sentiment classification will be divided into 3 categories, namely positive, negative, and neutral. Based on experiments in building a sentiment analysis model using 1D-CNN as many as 30 experiments, the best model was found in analyzing sentiment with an accuracy of 80.22%.


## 🎈 Usage <a name="usage"></a>

You can access the "App predict" folder and run the app.py file to see the results. Prediction result file will be saved in "App Predict/output" folder

## 🚀 Deployment <a name = "deployment"></a>

1. File Crawling_Selenium_BeautifulSoup.ipynb used to collect data on the female daily review website.
2. After the data is collected, it is necessary to preprocess the data which consists of the following stages (in Preprocessing.ipynb file): 
a. Data cleansing (remove duplicate data, punctuation), 
b. Case folding (equalizing the shape of letters), 
c. Normalization (changing words to standard forms), 
d. Stemming (changing to basic word forms), and 
e. Tokenization (creating a vocabulary list based on collected reviews).
3. For the labeling process using existing research conducted by Wahid and his colleagues
4. Then the word embedding process utilizes the research that has been done by Dery using word2vec.
5. Experimental stage for build the prediction model contained in the word2vec_dan_CNN.ipynb
6. The results of the model that has been built will then be saved and implemented into a website. 

## ⛏️ Built Using <a name = "built_using"></a>

- Selenium, ChromeDriver Manager and BeautifulSoup - Web Crawling
- Excel CSV file - Dataset
- TensorFlow - Machine Learning Framework
- Flask - Micro Web Framework
- Python - Programming Language

## ✍️ Authors <a name = "authors"></a>

- [@deviolettah](https://github.com/deviolettah)


