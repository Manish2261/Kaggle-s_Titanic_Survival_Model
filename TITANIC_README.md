# Kaggle-s-Titanic-Survival-Prediction-Model
It is a Kaggle Introductory competition aimed at supporting the learning path of the ML Enthusiast or a professional to apply Regression 
Techniques learnt thereon in the courses. It is the first step into these ver expanding world of machine Learning.


## Details regarding the competition are as follows:

### The Competition:
The sinking of the Titanic is one of the most infamous shipwrecks in history of mankind.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg.
Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.While 
there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. In this
challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using 
passenger data (ie name, age, gender, socio-economic class, etc).

## Data that I will be using in these competition:

In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic 
class, etc. One dataset is titled `train.csv` and the other is titled `test.csv`.

Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they 
survived or not, also known as the “ground truth”.

The `test.csv` dataset contains similar information but does not disclose the “ground truth” for each passenger. It’syour job to predict 
these outcomes.

Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

## Goals of Particpating:

We are demanded to submit the prediction result for the test data that we got in terms of 1 or 0 i.e(Survived or Not Survived) respectively therein  in the wrecking incident. These results of us are validated by the Ground truth results of the incident and the accuracy score are 
there conferred to our submission.

## Details regarding the datasets:

Different variable with their significance explained below:

Survival : ( 1 or  0) int64
It is an expected output from the test_set provided therein but our provided in the Training_set for the purpose of out=r model training.
The value are int64 type with value 1 or 0 for Survuved or Not Survived.

pclass: 'Categorical'
It represents the socio economic class for the passengers which needs to be evalluated judiciously as they may dictate the prediction results for passengers survival. 
They are int64 with values 1 - 1st class ticket, 2 - 2nd class ticket, 3 - 3rd class ticket

sex: 'Categorical'
It is a categorical feature which may be useful to determine the corelation between the survival rate between male and female.
It is having values as 'Male' and 'Female'

age: int64
It is an int64 giving tha age value

sibsp : int64
It signifies the presence of any siblings on the wrecked ship and their relation with survival rate.

parch: int64
It signifies the presence of parents or children(off-springs) of the passengers traveling on the ship

ticket :  int64
It signifies the ticket number of passengers consisting of the null values which needs to be fixed therein.

fare : int64
It signifies the ticket price of the passengers

cabin: int64
It gives the cabin number of the embarked passengers on the ship

embarked : 'Categorical'
It is a categorical variable featuring the boarding point or location of the passenger.
It is having values:
C - Cherbourg
Q - Queenstown
S - Southampton


