|                                                                                                                                          Question                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                         **1. Which linear regression algorithm can you use if you have a training set with millions of features?**                                                                                          |                                                                                                                                                                                                                                                                                                                                                                              Basically any Gradient Descent algorithm is better suited for cases where there is a large number of features. For a spead-up (to avoid computing the gradients on the entire training set at every step) Stochastic Gradient Descent or Mini-Batch Gradient Descent (when hardware accelration is available) could be an optimal solution. .                                                                                                                                                                                                                                                                                                                                                                              |
|                                                                  **2. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?**                                                                   |                                                                                                                                                                                        For the scope of this chapter, all linear algorithms, regularized or not will suffer from features having very different scales. For Gradient Descent algorithms it has a consequences of increasing the time it takes for finding global minimum (for the cost function). For regression models it will result in prediction precision loss. As outlined previously in Chapter 2, the optimal way to address this is to prepare your data in advance to avoid biases, removing outliers. This could, for example, be achieved by taking a `log` of your numerical values, or a square root. This is called scaling, and libraries often ship different flavors of scalers, such as `StandardScaler` from `Scikit-Learn`.                                                                                                                                                                                        |
|                                                                                             **3. Can gradient descent get sutck in a local minimum when training a logistic regression model?**                                                                                             |                                                                                                                                                                                                                                                                                                                                                     It can. Choice of the cost function implementation is crucial here. Potentially if a cost function is not regular, i.e. it contains plateaus ridges, etc, then potentially there is a risk that the algorithm will converge to a _local minimum_. For MSE cost function this should not be the case though, which is a _convex function_. There is no local minimum, just one _global minimum_.                                                                                                                                                                                                                                                                                                                                                     |
|                                                                                          **4. Do all gradient descent algorithms lead to the same model, provided you let them run long enough?**                                                                                           | Let's start by disambiguating and saying that under term _model_ we will understand a _type of model_ as well as its parameter vector: the bias term and feature weights. The idea behind all GD algorithms is similar: tweak parameters iteratively in order to minimize a cost function. What differs is the number of those iterations that it takes to converge. This is determined by _learning rate_ hyperparameter (size of steps). Using a _convex function_, such as MSE, as previously mentioned guarantees that the _global minimum_ will be found, but there will be still specifics. For example, _Batch Gradient Descent_ might potentially get further away from the global minimum if learning rate is too high. _Stochastic Gradient Descent_ will end-up close to the minimum, but it may continue to bounce around, never settling down. Due to the specifics of its implementation, _Mini-Batch Gradient Descent_ works on a random subset of the training set, but it will less likely escape _global minimum_ and the bouncing around will be smaller than for the stochastic GD. |
|                                        **5. Suppose you use batch gradient descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?**                                         |                                                                                                                                    This may indicate that the model has started to overfit the training data. One solution is to stop after the validation error has been above the minimum for some time and then roll back the model parameters to the point where the validation error was at minimum. Another possibility could be that the learning rate is too high, getting further and further away from the solution. To find a good learning rate, we could use a grid search. You may want to limit the number of epochs so that grid search can eliminate models that take too long to converge. To set an optimal number of epochs, you may want to try to set a very large number of epocs but to interrupt the algorithm when the gradient vector becomes tiny - that is, when the norm becomes smaller than a tiny number € (called the _tolerance_)                                                                                                                                    |
|                                                                                         **6. Is it a good idea to stop mini-batch gradient descent immediately when the validation error goes up?**                                                                                         |                                                                                                                                                                                                                                                                                                                                                               Error rate may jump around, due to the algorithm's random character. Instead of stopping at the first error, a valid alternative could be to use a good learning schedule: start with a large learning rate, then gradually reduce it, allowing the algorithm to settle at the global minimum. Save your models at some interval(s) and revert to the best model as needed.                                                                                                                                                                                                                                                                                                                                                               |
|                                         **7. Which gradient descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?**                                         |                                                                                                                                                                                                                     Due to the nature of Stochastic GD, that works only on a single instance at a time, the algorithm works much faster and will likely reach the vicinity of global minimum, but, unlike with other GD algorithms it will continue to bounce around, so it may take Stochastic GD more time to actually converge (as compared to Batch GD, for example). One thing that could help with that is to shuffle the instances during training. Because of that Batch GD is likely to end up near minimum (at the expense of more time spent). Another factor that can improve convergence for both Stochastic GD and Mini-Batch GD is to use a good learning schedule.                                                                                                                                                                                                                      |
|                            **8. Suppose you are using polynomial regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?**                             |                                                                                                                                                                           Typically this is indicative of an overfitting scenario. One way to improve an overfitting model is to feed it more training data until the validation error reaches the training error. The three ways refer to the **Bias/Variance** trade off: Bias, Variance and Irreducible data. Bias is more characteristic of an underfitting scenario. In our case we are likely whitnessing model's excessive sensitivity to small variations and the way to fix this is to constrain the model (i.e. to _regularize_ it). In the context of polynomial regression, constraining means reducing the number of polynomial degrees. Another part of the trade off is the noisiness of the data itself. The only way to reduce this is to clean up the data.                                                                                                                                                                           |
| **9. Suppose you are using ridge regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?** |                                                                                                                                                                                                                                                                                                                                                                             The described symptoms point to an underfitting problem. Bias is the part of the generalization error which is made due to wrong assumptions, such as the data is linear when it is actually quadratic. To make your model more reasonable, the recommended approach is to **decrease**  α, thus increasing the variance and reducing the bias.                                                                                                                                                                                                                                                                                                                                                                             |
|                                 **10. Why would you want to use: a. _Ridge regression_ instead of plain _linear regression_ (i.e. without any regularization)? b. _Lasso_ instead of _ridge regression_? c. _Elastic net_ instead of _lasso regression_?**                                  |                                            **a.** _Linear regression_ is good for linear data. Even with regularization hyperparameter set to 0, _ridge regression_ will do a better job at trying to shape data. **b.** Lasso and Ridge differ primarily in the _norm_ that they use for their _regularization term_. Ridge regression uses _l<sub>2</sub>_, whereas Lasso regression uses _l<sub>1</sub>_. Generally, the higher the norm (the subscript sequence number) index, the more it focuses on large values and neglects small ones. This is why _RMSE_ is more sensitive to outliers than the _MAE_. By consequence, Lasso tends to eliminate the weights of the least important features, by setting them to zero. So, Ridge is a good default, but if you suspect that only a few features are useful, you should prefer lasso. **c.** Elastic net is preferred over Lasso because Lasso may behave erratically when the number of features is greater than the number of traininh instances or when several features are strongly correlated.                                            |
|                                                    **11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two logistic regression classifiers or one softmax regression classifier?**                                                    |                                                                                                                                                                                                                                                                                                                                                                                              Softmax regression should be used only with mutually exclusive classes, here the task is to classify pictures that are likely to be combination of two classes: daytime outdoor, or nighttime indoors. For this specific case, training the dataset should be done using two logistic regression classifiers.                                                                                                                                                                                                                                                                                                                                                                                              |
|                                                  **12. Implement batch gradient descent with early stopping for softmax regression without using `Scikit-Learn`, only `NumPy`. Use it on a classification task such as the iris dataset.**                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |