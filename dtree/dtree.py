#Classifing in male or female using decision tree - https://www.youtube.com/watch?v=T5pRlIbr6gg


from sklearn import tree

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)


#test the classifier:
prediction = clf.predict([[190, 70, 43], [158, 62, 37]])


# CHALLENGE compare their results and print the best one!

print('DecisionTreeClassifier for gender according to height, weight and shoe_size')
print('The classification is: {}'.format(prediction))
