from django.shortcuts import render
#import all librires
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
# univariate cnn example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

def index(request):
    return render(request,'index.html')
def predict(request):
    return render(request,'predict.html')
def result(request):
	varity=request.POST["varity"]
	state=request.POST["state"]
	dist=request.POST["dist"]
	year=request.POST["year"]
	month=request.POST["month"]
	found=0
	pathP="Predicted/"+varity+"/"+state+"/"+dist+".csv"
	if os.path.isfile(pathP):
		print("file found")
		found=1
	else:
		pathD="Data/"+varity+"/"+state+"/"+dist+".csv"
		print(pathD)
		df=pd.read_csv(pathD)
	print("Hi Siddharoodha")
	#print(os.getcwd())
	#print(pd.read_csv(path))
	if found==1:
		print(pathP)
		df=pd.read_csv(pathP)
	else:
		df.drop(columns=['Unnamed: 0'],inplace=True)
		df=df.set_index("date")
		#create a new data frame with only "model price  coln"
		data=df.filter(['model price'])
		#visualize the closing history
		plt.figure(figsize=(16,8))
		plt.title(df["District"][0])
		plt.plot(df["model price"])
		plt.xlabel("Date",fontsize=18)
		plt.ylabel("model price(Rs/Quintal) ",fontsize=18)
		plt.legend(['model price','Arrival'],loc="lower right")
		plt.show()
		#convert the dataframe to a numpy a
		dataset=data.values
		trainig_data_len=math.ceil(len(dataset)*.8)
		scalar=MinMaxScaler(feature_range=(0,1))
		scaled_data=scalar.fit_transform(dataset)
		train_data=scaled_data[0:trainig_data_len,:]
		#create the tarining data set for model price
		#create scalar trainig data set
		train_data=scaled_data[0:trainig_data_len,:]
		#split the data into x and y data
		x_train=[]
		y_train=[]
		for i in range(12,len(train_data)):
			x_train.append(train_data[i-12:i,0])
			y_train.append(train_data[i,0])
		#convetr x_train and y_train to numpy  for model price
		x_train,y_train=np.array(x_train),np.array(y_train)
		x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

		#build the CNN model for model  price
		# define cnn model for model price
		model = Sequential()
		model.add(Conv1D(filters=64, kernel_size=2, activation='relu',input_shape=(x_train.shape[1],1)))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Flatten())
		model.add(Dense(50, activation='relu'))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
		#for model price
		model.fit(x_train,y_train,batch_size=1,epochs=50,verbose=1)

		#for feature predection dataset for model price
		test_data=scaled_data[trainig_data_len+11:,:]
		x_test=[]
		for i in range(1,len(test_data)):
			x_test.append(test_data[i])
		y=np.array(x_test)
		x_test=[]
		x_test.append(y)

		x_test
		feat=[]
		x_test=np.array(x_test)
		x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
		for i in range(120):#for two years from 2018 to
			#get the models predicted price values
			predictions=model.predict(x_test)
			for i in range(11):
				x_test[0][i][0]=x_test[0][i+1][0]
			x_test[0][11][0]=predictions
			predications=scalar.inverse_transform(predictions)
			feat.append(predications)
		D=[]
		for i in range(10):
			y=str(2018+i)
			for i in range(1,13):
				if i<10:
					m="0"+str(i)
				else:
					m=str(i)
				d="01"
				D.append(y+'-'+m+'-'+d)
		dict = {'Date':D}
		df = pd.DataFrame(dict) 
		df["feature"]=feat
		df['year'] = pd.DatetimeIndex(df['Date']).year
		df['month'] = pd.DatetimeIndex(df['Date']).month
		df.to_csv(pathP)




		



		


	print(df)
	#print(pd.read_csv(patH))

	df=df[df["year"]==int(year)]
	val=df[df["month"]==int(month)]
	
	context={"varity":varity,"state":state,"dist":dist,"year":year,"month":month,"result":val["feature"].values[0]}
	return render(request,'result.html',context)