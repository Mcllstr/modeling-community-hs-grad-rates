## UNDER CONSTRUCTION
# Predicting high school graduation rates

### Contributors
Joe McAllister
Tim Christy

### Abstract

Using Data from countyrankings.org we created multiple models to predicts high school performance based on an array of county-wide social economic data.  This analysis highlights other factors that appear to correlate with graduation rates and have heavy weights in the models.  

Graduation rates were binned based on quartiles and treated as multiclass classifier targets.  The upper quartile, median, and lower graduation rate quartiles were 93%, 89%, and 84%.  

A K-Nearest Neighbors classifier achieved a 45% prediction accuracy.  Trials with SVMs, logistic regression, random forests, and boosted forest methods yielded similar results.  While the overall accuracy is low, this is a significant improvement over the baseline random chance guess accuracy of 25%.  A quick examination of the confusion matrix below demonstrates that the model is good at accuratley predicting extremes above the upper quartile of 93% and below the lower quartile of 84%, but struggles when it comes to correct classifications between the upper and lower quartiles.  

![Multiclass KNN Confusion Matrix](/images/knn_conf_matrix.jpg)

Flattening the groupings into 3 equal size classes rather than 4 would likely improve the overall accuracy of the model.

The features that correlated most (postively and negatively) with high school graduation rates appear to mirror longstanding social, economic, and racial disparities.  Burdensome housing costs relative to income had the most significant correlation, and unsurprisingly had a very negative affect on graduation rates.  This may be a better predictor for economic disadvantages than median income or even the percentage of children in poverty - while there is one federal poverty line, costs of living vary drastically from state to state and city to city.  A person living in New York City whose income above the poverty line is likely be poorer relative to someone making an equal income in a lower cost of living city like Columbus Ohio.  

![Features that correlate highly with HS grad rates](/images/correlations.jpg)


### Files
- technical_workbook.ipynb
- slides.pdf
- README.md
- /images
- /data
