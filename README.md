# Modeling high school graduation rates

### Contributors
- Joe McAllister
- Tim Christy

### Abstract
We created multiple models to predict high school performance based on an array of county-wide socio-economic data.  This analysis highlights other factors that This analysis highlights other factors that appear to correlate with graduation rates and have heavy weights in the models.  

Graduation rates were binned based on quartiles and treated as multiclass classifier targets.  The upper quartile, median, and lower graduation rate quartiles were 93%, 89%, and 84%.  

An ensemble classifier including KNN, SVM, Logistic Regression, XGBoost, and Random Forest achieved the highest accuracy of 49%, this is a significant improvement over baseline random chance accuracy of 25%.  

A K-Nearest Neighbors classifier achieved a 45% prediction accuracy.  Trials with SVMs, logistic regression, random forests, and boosted forest methods yielded similar results.  While the overall accuracy is low, this is a significant improvement over the baseline random chance guess accuracy of 25%.  

![Ensemble Classifier Confusion Matrix](/images/logr_gs_matrix.jpg)

A quick examination of the confusion matrix below demonstrates that the model is good at accurately predicting extremes in the upper quartile above 93% and in the lower quartile below 85%, but struggles when it comes to correct classifications between the upper and lower quartiles.  

Exploring this further we can look at the feature importances for the logistic classifier.  The logistic classifier displays similar characteristics to the ensemble method, predicting extreme high and lower quartile well, but struggling to find sufficient differentiation between middle and upper quartiles.  The plot below shows the relative importance of weights in the regression model, the size of each circle and its color indicate the level of influence the variable has on the probability of a community belonging to any one of the quartiles.

![Logistic Classifier Feature Importances](/images/feature_importance_logistic_classifier_heat_cirlces.jpg)

Flattening the groupings into 3 equal size classes rather than 4 would likely improve the overall accuracy of the model.

The features that correlated most (postively and negatively) with high school graduation rates appear to mirror longstanding social, economic, and racial disparities.  Burdensome housing costs relative to income had the most significant correlation, and unsurprisingly had a very negative affect on graduation rates.  This may be a better predictor for economic disadvantages than median income or even the percentage of children in poverty - while there is one federal poverty line, costs of living vary drastically from state to state and city to city.  A person living in New York City whose income above the poverty line is likely be poorer relative to someone making an equal income in a lower cost of living city like Columbus Ohio.  

![Features that correlate highly with HS grad rates](/images/correlations.JPG)


### Files
- technical_workbook.ipynb
- slides.pdf
- README.md
- /images
- /data
