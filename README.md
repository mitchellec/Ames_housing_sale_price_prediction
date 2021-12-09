# Project 2: Ames Housing Data and Kaggle Challenge

# Ames Housing Sale Price prediction

__Background__<br>
For our second project, we are going to take a look at the Ames Housing Data and submit our sale price prediction in [Kaggle](https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge).  


__Contents__
1. [Problem Statement](#problem-statement)
2. [Executive Summary](#executive-summary)
3. [Data Sources](#data-sources)
4. [Data Dictionary](#data-dictionary)
5. [Python Library Used](#python-library-used)
6. [EDA](#exploratory-data-analysis)
7. [Preprocessing and Modeling](#preprocessing-and-modeling)
8. [External Research](#external-research)
9. [Conclusions and Recommendations](#conclusions-and-recommendations)
10. [References](#ref)
---

## Problem Statement


### Kaggle Challenge

As mentioned in the background, the goal of this project is to predict the sales price for each house. For each Id in the test set, we needs to predict the value of the `SalePrice` variable.

The challenge evaluation is determined by [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE).

There was no problem statement included in the Kaggle Challenge, hence we had developed the following Problem Statement to give a context to this project.

### Problem Statement developed
A real estate firm is looking to invest in Housing in Ames, Iowa and has obtained the selling price of some given homes from 2006 to 2010. However, some of the houses do not have sale price to compare against.

As data scientist working for the real estate firm, we were tasked to develop a model to predict the selling price of a given home in Ames, Iowa from 2006 to 2010, and predict the missing Sale Price in our dataset.

Our employer hopes to use this information to help assess whether the asking price of a house is higher or lower than the true value of the house. If the house is undervalued, it may be a good investment for the firm.

After investing, our employer would also like to find out what features drives the price of a house located in Ames, Iowa as potential redevelopment possibilities to increase the sale price.

__Summary of problem statement__<br>
- To explore and analyse the dataset to develop a model (with lowest RMSE) that predicts the housing sale price in Ames, Iowa.
- Using the model built, identify the top 3 features that will increase the sale price and identify 3 features that will lead to a decrease in the sale price.

__Report__<br>
This report is prepared for our management as we report back on the task assigned to us.

---

## Executive summary
### Ames, Iowa, United States
Ames is a city in Story County, Iowa, United States. It is home to Iowa State University (ISU), with leading agriculture, design, engineering and veterinary medicine colleges. ([source](https://en.wikipedia.org/wiki/Ames,_Iowa))

In 2015, Ames was named one of the “15 Cities That Have Done the Best Since the Recession” by Bloomberg Business and one of the top 25 “Best Places for STEM Grads.” ([source](https://www.cityofames.org/))

### Summary of Analysis
We will be using the data that had the features and Sale Price to see the spread of the data points against Sale Price. We will also be creating Regression Models using:
1. Linear Regression;
2. Ridge Regression;
3. Lasso Regression; and
4. Elastic Net Regression, which is a combines Ridge and Lasso penalties.

From the models created, we will take the model that had the lowest RMSE score to predict the Sale Price of the houses in the test set.

### Kaggle Challenge
As this project includes a Kaggle Challenge, we will upload the predicted sale price to the Kaggle Competition to see how our model prediction fares.

---

## Data Sources
- Data files: [Kaggle](https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge)
  1. train.csv - Data to be used for training
  2. test.csv - Data to be used for predicting sale price
  3. sample_sub_reg.csv - File format for submission in Kaggle
- Data description: [Source](https://web.archive.org/web/20201203235151/http://jse.amstat.org/v19n3/Decock/DataDocumentation.txt)
  - A copy of the source description can be found [here](../datasets/data_description.md)

---

## Data Dictionary

|	Feature	|	Type	|	Dataset	|	Description	|	Data Cleaning (if any)	|
|	-	|	-	|	-	|	-	|	-	|
|	Id	|	 int64	|	train_data, test_data	|	Observation number	|		|
|	PID	|	 int64	|	train_data, test_data	|	Parcel identification number - can be used with city web site for parcel review.	|		|
|	MS_SubClass	|	 int64	|	train_data, test_data	|	Identifies the type of dwelling involved in the sale.	|		|
|	MS_Zoning	|	 object 	|	train_data, test_data	|	Identifies the general zoning classification of the sale.	|		|
|	Lot_Frontage	|	 float64	|	train_data, test_data	|	Linear feet of street connected to property	|	Removed due to more than 5% of missing data	|
|	Lot_Area	|	 int64	|	train_data, test_data	|	Lot size in square feet	|		|
|	Street	|	 object 	|	train_data, test_data	|	Type of road access to property	|		|
|	Alley	|	 object 	|	train_data, test_data	|	Type of alley access to property	|		|
|	Lot_Shape	|	 object 	|	train_data, test_data	|	General shape of property	|		|
|	Land_Contour	|	 object 	|	train_data, test_data	|	Flatness of the property	|		|
|	Utilities	|	 object 	|	train_data, test_data	|	Type of utilities available	|		|
|	Lot_Config	|	 object 	|	train_data, test_data	|	Lot configuration	|		|
|	Land_Slope	|	 object 	|	train_data, test_data	|	Slope of property	|		|
|	Neighborhood	|	 object 	|	train_data, test_data	|	Physical locations within Ames city limits	|		|
|	Condition_1	|	 object 	|	train_data, test_data	|	Proximity to various conditions	|		|
|	Condition_2	|	 object 	|	train_data, test_data	|	Proximity to various conditions (if more than one is present)	|		|
|	Bldg_Type	|	 object 	|	train_data, test_data	|	Type of dwelling	|		|
|	House_Style	|	 object 	|	train_data, test_data	|	Style of dwelling	|		|
|	Overall_Qual	|	 int64	|	train_data, test_data	|	Rates the overall material and finish of the house	|		|
|	Overall_Cond	|	 int64	|	train_data, test_data	|	Rates the overall condition of the house	|		|
|	Year_Built	|	 int64	|	train_data, test_data	|	Original construction date	|		|
|	Year_Remod/Add	|	 int64	|	train_data, test_data	|	Remodel date (same as construction date if no remodeling or additions)	|		|
|	Roof_Style	|	 object 	|	train_data, test_data	|	Type of roof	|		|
|	Roof_Matl	|	 object 	|	train_data, test_data	|	Roof material	|		|
|	Exterior_1st	|	 object 	|	train_data, test_data	|	Exterior covering on house	|		|
|	Exterior_2nd	|	 object 	|	train_data, test_data	|	Exterior covering on house (if more than one material)	|		|
|	Mas_Vnr_Type	|	 object 	|	train_data, test_data	|	Masonry veneer type	|	Fill with highest count of non-null values	|
|	Mas_Vnr_Area	|	 float64	|	train_data, test_data	|	Masonry veneer area in square feet	|	To fill the missing values based on Mas Vnr Type	|
|	Exter_Qual	|	 object 	|	train_data, test_data	|	Evaluates the quality of the material on the exterior	|		|
|	Exter_Cond	|	 object 	|	train_data, test_data	|	Evaluates the present condition of the material on the exterior	|		|
|	Foundation	|	 object 	|	train_data, test_data	|	Type of foundation	|		|
|	Bsmt_Qual	|	 object 	|	train_data, test_data	|	Evaluates the height of the basement	|	Fill with highest count of non-null values	|
|	Bsmt_Cond	|	 object 	|	train_data, test_data	|	Evaluates the general condition of the basement	|	Fill with highest count of non-null values	|
|	Bsmt_Exposure	|	 object 	|	train_data, test_data	|	Refers to walkout or garden level walls	|	Fill with highest count of non-null values	|
|	BsmtFin_Type_1	|	 object 	|	train_data, test_data	|	Rating of basement finished area	|	Fill with highest count of non-null values	|
|	BsmtFin_SF_1	|	 float64 / int64	|	train_data, test_data	|	Type 1 finished square feet	|	Fill with highest count of non-null values. However, as `BsmtFin Type 1` was filled with value, we filled with the highest count of non-zero value	|
|	BsmtFin_Type_2	|	 object 	|	train_data, test_data	|	Rating of basement finished area (if multiple types)	|	Fill with highest count of non-null values	|
|	BsmtFin_SF_2	|	 float64 / int64	|	train_data, test_data	|	Type 2 finished square feet	|	Fill with highest count of non-null values	|
|	Bsmt_Unf_SF	|	 float64 / int64	|	train_data, test_data	|	Unfinished square feet of basement area	|	Fill with highest count of non-null values	|
|	Total_Bsmt_SF	|	 float64 / int64	|	train_data, test_data	|	Total square feet of basement area	|	Fill with sum of `BsmtFin_SF_1`, `BsmtFin_SF_2` and `Bsmt_Unf_SF`	|
|	Heating	|	 object 	|	train_data, test_data	|	Type of heating	|		|
|	Heating_QC	|	 object 	|	train_data, test_data	|	Heating quality and condition	|		|
|	Central_Air	|	 object 	|	train_data, test_data	|	Central air conditioning	|		|
|	Electrical	|	 object 	|	train_data, test_data	|	Electrical system	|	Fill with highest count of non-null values	|
|	1st_Flr_SF	|	 int64	|	train_data, test_data	|	First Floor square feet	|		|
|	2nd_Flr_SF	|	 int64	|	train_data, test_data	|	Second floor square feet	|		|
|	Low_Qual_Fin_SF	|	 int64	|	train_data, test_data	|	Low quality finished square feet (all floors)	|		|
|	Gr_Liv_Area	|	 int64	|	train_data, test_data	|	Above grade (ground) living area square feet	|		|
|	Bsmt_Full_Bath	|	 float64 / int64	|	train_data, test_data	|	Basement full bathrooms	|		|
|	Bsmt_Half_Bath	|	 float64 / int64	|	train_data, test_data	|	Basement half bathrooms	|		|
|	Full_Bath	|	 int64	|	train_data, test_data	|	Full bathrooms above grade	|		|
|	Half_Bath	|	 int64	|	train_data, test_data	|	Half baths above grade	|		|
|	Bedroom_AbvGr	|	 int64	|	train_data, test_data	|	Bedrooms above grade (does NOT include basement bedrooms)	|		|
|	Kitchen_AbvGr	|	 int64	|	train_data, test_data	|	Kitchens above grade	|		|
|	Kitchen_Qual	|	 object 	|	train_data, test_data	|	Kitchen quality	|		|
|	TotRms_AbvGrd	|	 int64	|	train_data, test_data	|	Total rooms above grade (does not include bathrooms)	|		|
|	Functional	|	 object 	|	train_data, test_data	|	Home functionality (Assume typical unless deductions are warranted)	|		|
|	Fireplaces	|	 int64	|	train_data, test_data	|	Number of fireplaces	|		|
|	Fireplace_Qu	|	 object 	|	train_data, test_data	|	Fireplace quality	|		|
|	Garage_Type	|	 object 	|	train_data, test_data	|	Garage location	|		|
|	Garage_Yr_Blt	|	 float64	|	train_data, test_data	|	Year garage was built	|	Removed due to more than 5% of missing data in train_data	|
|	Garage_Finish	|	 object 	|	train_data, test_data	|	Interior finish of the garage	|	Fill with highest count of non-null values based on `Garage Type`	|
|	Garage_Cars	|	 float64 / int64	|	train_data, test_data	|	Size of garage in car capacity	|	Fill with highest count of non-null values based on `Garage Type`	|
|	Garage_Area	|	 float64 / int64	|	train_data, test_data	|	Size of garage in square feet	|	Fill with highest count of non-null values based on `Garage Type`	|
|	Garage_Qual	|	 object 	|	train_data, test_data	|	Garage quality	|	Fill with highest count of non-null values based on `Garage Type`	|
|	Garage_Cond	|	 object 	|	train_data, test_data	|	Garage condition	|	Fill with highest count of non-null values based on `Garage Type`	|
|	Paved_Drive	|	 object 	|	train_data, test_data	|	Paved driveway	|		|
|	Wood_Deck_SF	|	 int64	|	train_data, test_data	|	Wood deck area in square feet	|		|
|	Open_Porch_SF	|	 int64	|	train_data, test_data	|	Open porch area in square feet	|		|
|	Enclosed_Porch	|	 int64	|	train_data, test_data	|	Enclosed porch area in square feet	|		|
|	3Ssn_Porch	|	 int64	|	train_data, test_data	|	Three season porch area in square feet	|	Removed as majority of the data are of 0 values.	|
|	Screen_Porch	|	 int64	|	train_data, test_data	|	Screen porch area in square feet	|		|
|	Pool_Area	|	 int64	|	train_data, test_data	|	Pool area in square feet	|	Removed as majority of the data are of 0 values.	|
|	Pool_QC	|	 object 	|	train_data, test_data	|	Pool quality	|		|
|	Fence	|	 object 	|	train_data, test_data	|	Fence quality	|		|
|	Misc_Feature	|	 object 	|	train_data, test_data	|	Miscellaneous feature not covered in other categories	|	Removed as majority of the data are of 0 values.	|
|	Misc_Val	|	 int64	|	train_data, test_data	|	$Value of miscellaneous feature	|		|
|	Mo_Sold	|	 int64	|	train_data, test_data	|	Month Sold (MM)	|		|
|	Yr_Sold	|	 int64	|	train_data, test_data	|	Year Sold (YYYY)	|		|
|	Sale_Type	|	 object 	|	train_data, test_data	|	Type of sale	|		|
|	SalePrice	|	 int64	|	train_data	|	Sale price $$	|		|
|	Total_house_SF	|	float64	|	train_data, test_data	|	Sum of `Total_Bsmt_SF`, `1st_Flr_SF`, `2nd_Flr_SF`, `Wood_Deck_SF`, `Open_Porch_SF`, `Enclosed_Porch` and `Screen_Porch` 	|	Feature Engineering	|
|	house_full_bath	|	float64	|	train_data, test_data	|	Sum of `Bsmt_Full_Bath` and `Full_Bath`	|	Feature Engineering	|
|	house_half_bath	|	float64	|	train_data, test_data	|	Sum of `Bsmt_Half_Bath` and `Half_Bath`	|	Feature Engineering	|
|	house_age	|	float64	|	train_data, test_data	|	Subtract `Year_Built` from `Yr_Sold`	|	Feature Engineering	|
|	remod_house_age	|	float64	|	train_data, test_data	|	Subtract `Year_Remod/Add` from `Yr_Sold`	|	Feature Engineering	|

---

## Python Library Used

Combining the python library used in both notebooks:

__For Calculations and Data Manipulations__
1. Numpy
2. Pandas

__For Graph Plottings__
1. Matplotlib
2. Seaborn

__For Modelling__
1. From sklearn
  1. metrics
  2. linear_model
    1. LinearRegression
    2. Ridge
    3. RidgeCV
    4. Lasso
    5. LassoCV
    6. ElasticNet
    7. ElasticNetCV
  3. model_selection
    1. train_test_split
    2. cross_val_score
    3. cross_val_predict
  4. preprocessing
    1. StandardScaler
    2. PolynomialFeatures

__For csv file exporting folder creation__
1. os

---

## Exploratory Data Analysis
In this section, we looked at the train dataset features' correlation with the sale price and picked the top 5 positive correlated features and top 3 negative correlated features (excluding `PID`) to look at the spread across the years using histogram with kde line and the quartiles using boxplot

Before starting any analysis of the data, each individual dataset was imported to a Pandas DataFrame and data cleaning was conducted to ensure all datatypes were accurate and any other errors identified were fixed. We also removed features that had either more than 5% of missing data and data that had outliers striking outliers observed from Lot_Area and Total_Bsmt_SF.

---

## Preprocessing and Modeling

### Preprocessing and Feature Engineering:
1. Perform Feature Engineering for Continuous features<br>
  - `Total_house_SF`: Sum of `Total_Bsmt_SF`, `1st_Flr_SF`, `2nd_Flr_SF`, `Wood_Deck_SF`, `Open_Porch_SF`, `Enclosed_Porch` and `Screen_Porch`
  - `house_full_bath`: Sum of `Bsmt_Full_Bath` and `Full_Bath`
  - `house_half_bath`: Sum of `Bsmt_Half_Bath` and `Half_Bath`
  - `house_age`: Subtract `Year_Built` from `Yr_Sold`
  - `remod_house_age`: Subtract `Year_Remod/Add` from `Yr_Sold`
2. Dummify categorical features using one-hot encoding <br>
  - First column to prevent multicollinearity

### Modeling:
Regression Models created:
1. Linear Regression;
2. Ridge Regression;
3. Lasso Regression; and
4. Elastic Net Regression, which is a combines Ridge and Lasso penalties.

---

## External Research

__Why do we log target variable?__
https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html


---

## Conclusions and Recommendations
### EDA
From our EDA performed, we noticed that we have features that have positive or negative correlation with the Sale Price.

__Features that had high positive correlation__<br>
- Overall_Qual: Rates the overall material and finish of the house
- Gr_Liv_Area: Above grade (ground) living area square feet
- Total_Bsmt_SF: Total square feet of basement area
- Garage_Area: Size of garage in square feet
- Garage_Cars: Size of garage in car capacity

__Features that had high negative correlation__<br>
- Enclosed_Porch: Enclosed porch area in square feet
- Kitchen_AbvGr: Kitchens above grade
- Overall_Cond: Rates the overall condition of the house

However, as correlation only looks at numeric features, we will not be able to interpret the relationship between categorical features and Sale Price.

For features that had high positive correlation, it made sense that they are there as buyers of houses tends to look out for such features and people tend to be willing to pay more if the house was maintained better or is bigger.

With this thinking, it would not make sense for the features that had negative correlation as people would assume higher (better) ratings of the house would mean there is an increase in the sale price.

### Modeling



### Kaggle Submission Score

Snapshot from kaggle:

<img src='./images/kaggle_submission.png' width=800px />

---

## References
1. Root Mean Square Error (RMSE) definition - https://en.wikipedia.org/wiki/Root-mean-square_deviation
2. Introduction of Ames, Iowa, United States - https://en.wikipedia.org/wiki/Ames,_Iowa
3. City of Ames - https://www.cityofames.org/
4. Log target variable - https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
---
