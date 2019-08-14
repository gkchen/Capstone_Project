# Job.search()
TDI Capstone project for 2019 summer session

Job.search() is a job location prediction engine that allows a user to input keywords associated with the job they are interested, and receive match probabilities for each US state, including Washington DC. This allows a user to, get an idea of the job market around the US at a glance, without extensive domain knowledge.

## Data

The data is provided by The Data Incubator and Thinknum consisting of 175 GB of job posts from companies listed on the New York Stock Exchange and NASDAQ, and US census data.

## Machine Learning

Job post titles were processed using gensim and scikit-learn natural language processing packages and TF-IDF vectorizer. The cleaned and vectorized job post data is used to iteratively train multinominal naive-bayes models with and without state population weights. To yield insights into the overall job market, an online SVD algorithm was used to decompose the job keywords into 50 sectors of industry.