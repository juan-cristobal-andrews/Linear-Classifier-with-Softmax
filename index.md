# Introduction
The idea of this notebook is to explore a step-by-step approach to create a <b>Linear Classifier with Softmax</b> without the help of any third party library. We will later compare its results with two other supervised learning algorithms such as Neural Networks and K-Nearest Neighbors in order to see if there's any difference in performance and accuracy.

In practice, these Algorithms should be useful enough for us to classify our data whenever we have already made clusters (in this case color) which will serve as a starting point to train our models.

## 1. Working Data

```R
# Load Data
SampleData <- read.csv("sample.csv")
SampleData$Class <- as.character(SampleData$Class)
```

```R
# Display all data
library(ggplot2)
colsdot <- c("1" = "blue", "2" = "darkred", "3" = "darkgreen")
ggplot() + 
  geom_point(data=SampleData,mapping=aes(x,y, colour=Class),size=3 ) +  
  scale_color_manual(values=colsdot) +
  xlab('X') + ylab('Y') + ggtitle('All Sample Data')
```

<img src="images/plot1.jpg" width="422" height="422" />

As we can observe, our data has 900 points distributed in the complex form of a spiral and it's classified in 3 clusters (Red, Green and Blue) en equal amounts (300 per class).

### 1.1 Train and test sample generation

We will create 2 different sample sets:
- <b>Training Set:</b> This will contain 75% of our working data, selected randomly. This set will be used to train our model.
- <b>Test Set:</b> Remaining 25% of our working data, which will be used to test the accuracy of our model. In other words, once our predictions of this 25% are made, will check the "<i>percentage of correct classifications</i>" by comparing predictions versus real values.

```R
# Training Dataset
smp_siz = floor(0.75*nrow(SampleData))
train_ind = sample(seq_len(nrow(SampleData)),size = smp_siz)
train =SampleData[train_ind,]

# Test Dataset
test=SampleData[-train_ind,]
OriginalTest <- test
```

### 1.2 Train Data

With this data we will generate train our models. This corresponds to 75% of our data.

```R
# Display data
library(ggplot2)
colsdot <- c("1" = "blue", "2" = "darkred", "3" = "darkgreen")
ggplot() + 
  geom_point(data=train,mapping=aes(x,y, colour=Class),size=3 ) +  
  scale_color_manual(values=colsdot) +
  xlab('X') + ylab('Y') + ggtitle('Train Data')
```

<img src="images/plot2.jpg" width="422" height="422" />

By comparing to the "all sample data" plot we can now observe much fewer points. These points will be used to train our algorithms into learning our training data classifications (RGB colors) as shown below.

### 1.3 Test Data

This corresponds to leftover (25%) data. Even though in this scenario we already know it's classification, we will simulate a more realistic case in which we don't, in order to "predict" it's colors.

```R
# Display data
library(ggplot2)
ggplot() + 
  geom_point(data=test,mapping=aes(x,y),size=3 ) +  
  scale_color_manual(values=colsdot) +
  xlab('X') + ylab('Y') + ggtitle('Test Data')
```

<img src="images/plot3.jpg" width="422" height="422" />

As seen in the plot above we now have many points missing since this only represents 25% of our training data. Additionally, our spiral is not very clear now and colors are completely gone. <b>We will try to predict these colors</b> by creating different algorithms which we will later compare in order to observe its accuracy.

## 2. Linear Classifier with Softmax

Below is a step-by-step example of a Linear Classifier with Softmax cost function.
What we want to achieve is for each selected gray point above (our test values), where we allegedly don't know it's true color, find the correct classification by finding the correct linear combination of characteristics in a very similar way on how we usually generate our lineal models.

<img src="images/formula.png" width="192" height="36" />

Note that our Weight (W) and Bias (b) are both Vectors, which dimensions of 2x3 and 1x3 respectively.

<b>Aditionally, we need to define: </b>
- <b>Score Function: </b> Our linear classifier.
- <b>Cost Function: </b> In this case we will use Softmax since it provides an intuitive output in the form of a normalized class probability. We will use cross-entropy loss.
- <b>Optimization Method</b>: We will use gradient descent in order to minimize our cost function.










