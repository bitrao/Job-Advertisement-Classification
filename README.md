# Job Advertisement Classification

This project aim to build an NLP model that can classify jobs into 4 categories: `Accounting_Finance`, `Engineering`, `Healthcare_Nursing` and `Sales` based on their **description** and/or **titles**. 

### Dataset
The dataset used to train the models is a collection of different job advertisements that are already divided into categories. 
The [data](../main/data) folder consists of four subfolders, *Accounting_Finance, Engineering, Healthcare_Nursing* and *Sales*. These subfolders describe the job categories, and each of the subfolders contains multiple job advertisements (text files) named "Job_XXXXX", where "XXXXX" is a unique number. 
A job advertisement has three parts: `Title`, `WebIndex` and `Description`. In some job advertisements, `Company` is also included. 
The `Title` is the title of the job, `WebIndex` is the unique digit identifier of the job advertisement, `Company`is the name of the company that looks for employees, and `Description` is the job description.

### Approach
The jobs information are preprocessed and represented in 4 differents features sets using `Bag of Words` model and also `words-embedding`. 
The bag-of-words model will generate a Count vector representation and TFIDF vector representation from the generated vocabulary, 
and for the word embedding model, we have chosen the Word2Vec pretrained model from Google News 300 to build a weighted and unweighted vector representation for the job descriptions. 
The weighted vector represention in this project is a Term Frequency-Inverse Document Frequency (TF-IDF) that checks how frequent a word appears in a job description relative to all the job advertisement descriptions. 

For the classifier, 2 algorithms are used: `Logistic Regression` and  `Support Vector Machine` (SVM).

# Feature Representation
### Count Features Vectors
<p align="center">
  <img src="https://github.com/tringuyenbao/Job-Advertisement-Classification/blob/main/images/tSNE-plot-for-count-vectors.png?raw=true" alt="tSNE-plot-for-count-vectors"/>
</p>

### TF-IDF Features Vectors
<p align="center">
  <img src="https://github.com/tringuyenbao/Job-Advertisement-Classification/blob/main/images/tSNE-plot-for-tfidf-vectors.png?raw=true" alt="categories-count"/>
</p>

### Unweighted word2vec vectors
<p align="center">
  <img src="https://github.com/tringuyenbao/Job-Advertisement-Classification/blob/main/images/tSNE-plot-for-unweighted-doc2vec.png?raw=true" alt="tSNE-plot-for-unweighted-doc2vec"/>
</p>

### TF-IDF Weighted word2vec vectors
<p align="center">
  <img src="https://github.com/tringuyenbao/Job-Advertisement-Classification/blob/main/images/tSNE-plot-for-tfidf-weighted-doc2vec.png?raw=true" alt="tSNE-plot-for-tfidf-weighted-doc2vec"/>
</p>

# Models

Since the dataset is small, models were trained using `LogisticRegression` and additionally, `Support Vector Machine` due to its ability to handle large feature space.

Models performance were evaluated using **5 folds cross validation** with 4 different sets of features presentation: `count vectors`, `tfidf vectors`, and additional `unweighted-doc2vec` and `tfidf-weighted-doc2vec` retrieved from word embeddings using **word2vec-google-300**.

Experiments on using the **titles representation** instead of the descriptions and using **both** were also conducted to see whether the models performance vary regardings the amount of data provided for training.

### Metrics
<p align="center">
  <img src="https://github.com/tringuyenbao/Job-Advertisement-Classification/blob/main/images/categories-count.png?raw=true" alt="categories-count"/>
</p>

The target variable is **not imbalanced**. It is adequate to use `accuracy score` as the primary metrics for models evaluation.

### Results

|Models                     |count    |tf-idf       |unweighted    |tfidf-weighted |
|---------------------------|---------|-------------|--------------|---------------|
|LR with descriptions       |0.871125 |0.885302     |0.818313      |0.853127       |
|SVM with descriptions      |0.854376 |**0.891762** |0.806708      |0.81962        |

Overall, `Logistic Regression` scores were better than `SVM` on most of the features representation except for the **tf-idf**, which was at 0.891 and also the highest average accuracy an approach could score on descriptions features. Both word-embedding features got lower scores than the other two, but as within those two only, the **tfidf weighted** set delivered better results for both models (0.853 for Logistic Regression and 0.819 for SVM). 


|Models                               |count                                    |tf-idf                                      |
|-------------------------------------|-----------------------------------------|--------------------------------------------|
|LR with descriptions                 |0.871125                                 |0.885302                                    |
|LR with titles                       |<span style="color:red">0.828619</span>  |<span style="color:red">0.840207</span>     |
|LR with both titles and descriptions |<span style="color:blue">0.881423</span> |<span style="color:blue">**0.893011**</span>|
|SVM with descriptions                |0.854376                                 |0.891762                                    |
|SVM with titles                      |<span style="color:red">0.805409</span>  |<span style="color:red">0.841481</span>     |
|SVM with both titles and descriptions|<span style="color:blue"> 0.871108</span>|<span style="color:red">0.882721</span>     |

By conducting the training process on the ***Titles***, the overall accuracy decreased significantly. Meanwhile, with the addition of ***Titles*** features, the results were slightly better. For `Logistic Regression` models the accuracy increased from 0.871 to 0.881 regarding count_features and from 0.885 to 0.893 regarding tfidf_features. Same with `SVM`, the score on count_features went up to 0.871 (by 0.02) but slightly decrease on tfidf_features. It can be concluded that with more data and information, we can considerably train more accurate models.

The best models is **Logistic Regression with tf-idf of Titles and Descriptions**

*Assignment 3 of COSC3015-Advanced Programming for Data Science RMIT 2024 (Group works with Nguyen Nguyen and Lisa Maria Huynh)*

