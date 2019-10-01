# Modeling high school graduation rates

### Contributors
- Joe McAllister
- Tim Christy

### Overview
We created multiple models to predict high school performance based on 500+ county-wide socio-economic variables extracted from countyrankings.org county health database.  These were widdled down to 64 variables, and these were used to model graduation rates and highlight other factors that  that appear to correlate with graduation rates and have significant influence in the models.  

Graduation rates were binned based on quartiles and treated as multiclass classifier targets.  The upper quartile, median, and lower graduation rate quartiles were 93%, 89%, and 84%.  

An ensemble classifier including KNN, SVM, Logistic Regression, XGBoost, and Random Forest achieved the highest accuracy of 49%, this is a significant improvement over baseline random chance accuracy of 25%.  

A quick examination of the confusion matrix below demonstrates that the model is good at accurately predicting extremes in the upper quartile above 93% and in the lower quartile below 85%, but struggles when it comes to correct classifications between the upper and lower quartiles.  

![Ensemble Classifier Confusion Matrix](/images/ensemble_gs_matrix.jpg)

Exploring this further we can look at the feature importances for a logistic classifier.  The logistic classifier displays similar characteristics to the ensemble method, predicting extreme highest and lowest quartiles well, but struggling to find sufficient differentiation between the middle quartiles.  

![Logistic Classifier Confusion Matrix](/images/logr_gs_matrix.jpg)

We can see this further reflected in the plot below.  This chart shows the relative importance of weights in the regression model, the size of each circle and its color indicate the level of influence the variable has on the odds of a community belonging to each of the quartiles.  The outer quartiles have stronger weights that help differentiate them from the other bins, where the weights in the inner two quartiles are relatively smaller and more homogenous in weight.  

![Logistic Classifier Feature Importances](/images/feature_importance_logistic_classifier_heat_cirlces.jpg)

The features that signaled most (postively and negatively) odds that a community has the highest or lowest graduation rates appear to mirror longstanding social, economic, and racial disparities.  The strongest predictor for the lowest graduation rate quartile were variables linked to access to healthcare (poor physical health, injury deaths), poverty (children eligible for meal assistance, burdensome housing costs), and race (higher % Native Americans).  While a higher percentage native americans signal increased odds for lower graduation rates, higher percentages of white people in a county signal reduced odds of having low graduation rates.  This is a stark reminder that native american communities face socio-economic disadvantages that whiter communities do not, and these diferences continue to harm future generations.  

![Lower Quartile Feature Importances](/images/lowest_quartile_feature_importance.jpg)

Somewhat surprisingly frequent physical distress is (at least in this model) the most the most significant contributor to odds of high graduation rate counties.  More investigation is needed to understand if this is a real relationship or some quirk of the data.  Percent hispanic, percent white population, and higher household incomes highly influence odds that a county fits into top graduation rate quartile.  Percent of single parent homes, higher incidence of mental illnesses, and unemployment reduce odds of high quartile graduation rates.  

![Upper Quartile Feature Importances](/images/highest_quartile_feature_importance.jpg)

### Conclusions
Our ensemble classifier achieved 49% percent accuracy in predicting county high school graduation rates quartiles, a gain over baseline random chance of 25%.  In the course of the modeling it was noted that many of the variables were heavily left or right skewed to different degrees, the model accuracy might be improved in the future by applying custom normalization functions to each of the 64 variables.  The influence of features on the model seem to mirror longstanding social, economic, and racial disparities, with whiteness and higher incomes tending to indicate higher odds of highest graduation rate class membership, and poverty and non-whiteness (and in particular native american communities) indicating higher odds of lowest graduation rate class membership.  These are a stark reminder that the playing field is not level for all kids in America.

### Files
- technical_workbook.ipynb
- slides.pdf
- README.md
- /images
- /data
