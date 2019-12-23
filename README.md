# Udacity Machine Learning Engineer Nanodegree
## Starbucks Capstone Project

### Problem Statement

The problem is simple: we want to make better purchasing offers to Starbucks' customers. For this, we can use customer's past behaviour to find patterns and try to be more assertive. As given by the Udacity's Starbucks Project Overview, the basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer. In other words, this is a classification problem where the model takes user behaviour data as input and produces a group as output (either previously defined or not).

This has been one of the [most used](https://towardsdatascience.com/how-to-predict-the-success-of-your-marketing-campaign-579fbb153a97) applications of machine learning in the industry, since it provides you with means to save money spent on marketing campaigns by directing content to users who are more likely to convert based on a multitude of characteristics. 

To evaluate the trained models, we'll compare the models based on it's F1-Score. This is a widely used metric to evaluate classification problems. [Aditya Mishra defines](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234) it as follows:
> F1 Score is the Harmonic Mean between precision and recall. The range for F1 Score is [0, 1]. It tells you how precise your classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances).

### Datasets and Inputs

As given by the Udacity's Starbucks Project Overview:

* The dataset comes from a program that simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.

* The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.

* Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.

* As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.

* There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.

The data is divided in 3 files:

**profile.json**: Rewards program users (17000 users x 5 fields)

* gender: (categorical) M, F, O, or null
* age: (numeric) missing value encoded as 118
* id: (string/hash)
* became_member_on: (date) format YYYYMMDD
* income: (numeric)

**portfolio.json**: Offers sent during 30-day test period (10 offers x 6 fields)

* reward: (numeric) money awarded for the amount spent
* channels: (list) web, email, mobile, social
* difficulty: (numeric) money required to be spent to receive reward
* duration: (numeric) time for offer to be open, in days
* offer_type: (string) bogo, discount, informational
* id: (string/hash)

**transcript.json**: Event log (306648 events x 4 fields)

* person: (string/hash)
* event: (string) offer received, offer viewed, transaction, offer completed
* value: (dictionary) different values depending on event type
* offer id: (string/hash) not associated with any "transaction"
* amount: (numeric) money spent in "transaction"
* reward: (numeric) money gained from "offer completed"
* time: (numeric) hours after start of test

The `event` data in `transcript.json` can be used as label for a supervised learning classification algorithm, to measure for success of the offer.

BOGO and discount offers are the inly ones with an associated `offer completed` event. Also, the `transaction` event doesn't have an associated id, which means that the join won't add data to those.

Basically, we can look into the BOGO and discount offers user funnels in this order:
* offer received
* offer viewed
* transaction
* offer completed

For the informational offer, the funnel has this order:
* offer received
* offer viewed
* transaction

With this, we were able to select features combined with the user profiles.

The final dataset had 13 columns of featues and 1 column regarding the label for success.

### Modeling
We used Scikit-learn's framework with cross-validation with 5 folds for all models. We also used train/test split with 80% for train and 20% as test.

Our independent variables - or features - are:
* reward
* difficulty
* duration
* email
* mobile
* social
* web
* amount
* age
* income
* became_member_on_year
* offer_type_encoded
* gender_encoded

We trained supervised models for this, where the `is_successful` is the predicted label. All models were compared by the F1-Score.

The supervised models trained, with respective F1-Scores, were:
* Baseline Model: Dummy Classifier for most frequent class
    * F1-Score: 64.41%
* Model 1: Logistic Regression
    * F1-Score: 66.62%
* Model 2: Naïve Bayes
    * F1-Score: 70.26%
* Model 3: Support Vector Machines (SVM)
    * 65.10%
* Model 4: Decision Tree
    * F1-Score: 65.08%

The best model was Naïve Bayes, but it has not explainability. The second best and explainable model was the Logistic Regression.

###  Conclusion
When we started, we wanted to make better purchasing offers to Starbucks’ customers. For
this, we used customer’s past behaviour to find patterns and try to be more assertive. As given
by the Udacity’s Starbucks Project Overview, the basic task was to use the data to identify which
groups of people are most responsive to each type of offer, and how best to present each type of
offer. In other words, this is a classification problem where the model takes user behaviour data as
input and produces a group as output (either previously defined or not).

For this project, we spent quite some time dealing with the features, manipulating tose to fit into the models. For that to happen, we found a way to define an offer success based on the user funnel performed from the transcript dataset.

Once we had the dataset, we trained 4 supervised learning models and a baseline one. The baseline was a incredibly naïve model that classified the items based on the most frequent class. The model with best performance was a Naïve Bayes. This model doesn't have easy explainability, which means we fail to find understainable patterns to provide offers.

In order to get explainability, we chose the Logistic Regression model, that has lower predictive power. But with this model, we identified the top three most important features: reward, social and mobile.

Next steps would include better feature engineering and selection (we just used all features we could) and other classification models. The model selection should probably account for model explainability, which failed in this case. 

## Instructions
1. Download the datasets
1. Run `Starbucks_Capstone_Project.ipynb`. Note that the files should be in the same directory

## Libraries
All used libraries can be found in the `requirements.txt` file.