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

The features that correlated most highly with high school graduation rates were
- 1
- 2
- 3
- 4
- 5 

### Files
- technical_workbook.ipynb
- slides.pdf
- README.md
- /images
- /data
