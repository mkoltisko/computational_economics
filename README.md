# Computational Economics
### Machine Learning to predict the price of diamonds 
Final project for ECON 380 at Case Western Reserve.

To achieve an R<sup>2</sup> value of 0.93, I drew from a consensus of Random Forest Regressions. 
The three most important categories (color, cut, and clarity) were qualitative and not strictly numeric. Instead of 
guessing at a numeric conversion, I separated the data to isolate that characteristic with its volumetric data,
ignoring all of qualities. Clustering together all of the diamonds which shared a specific characteristic, I then 
used a basic Random Forest Regression to correlate the price of the diamond with that specific quality and volume. 
When done for each diamond, three new columns were added for the predicted price based on the rank of their 
qualities. The final price prediction came from averaging the qualitative predictions.

Why take the average of predictions? Looking at the each category and each regression, which one comes the closest 
to the actual price?
It could be possible that a certain category is more important than another which could skew a consensus.
For example, the rank of clarity could heavily affect the final price such that a low clarity would depreciate the 
diamond's price even if it had a good color or cut. In this case, we should weigh the prediction based on clarity 
more heavily in the final price. From this exploration and pie chart, we can see that each qualitative prediction 
is closest to the correct price an even amount of the time. Therefore, they are all good indicators of the price and
the model should evenly weigh these three categories when calculating the final prediction.
