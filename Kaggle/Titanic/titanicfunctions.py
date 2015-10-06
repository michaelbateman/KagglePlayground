def cleandata(df): 	# input is a pandas dataframe called df
				# output is an array 


	print ' I am in the function'

	# Converting gender to 0 for female 1, for male

	df['Gender'] = df['Sex'].map( { 'female': 0, 'male': 1 } ).astype(int)
	
	
	# Filling in missing ages with median age for the correct gender/passenger class
	
	median_ages = np.zeros((2,3))
	for i in range(0,2):
		for j in range(0,3):
			median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1) ].Age.dropna().median()
	df['AgeFill'] = df['Age']
	print 'check 1'
	for i in range(0,2):
		for j in range(0,3):
			df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]


	# Converting point of embarcation into an integer

	df['Port'] = df['Embarked'].map( { 'C':0, 'S':1, 'Q':2})
	df[df['Embarked'].isnull()] = 0
	print df[df.Age.isnull()]
	# Dropping irrelevant columns
	
	#print df[df.Age.isnull()]
	print df.head(10)
	print df.info()
	print df[ df['Port'].isnull()]
	
	# filling in missing fares with median for the correct passenger class
	median_fare = [0,0,0]
	for j in range(0,3):
		median_fare[j] = df[df['Pclass'] == j+1].Fare.dropna().median()
	df['FareFill'] = df['Fare']
	for j in range(0,3):
		df.loc[ (df.Fare.isnull())  & (df.Pclass == j+1), 'FareFill'] = median_fare[j]
	
	
	df = df.drop(['Fare','Cabin', 'Name', 'Ticket','Age', 'Embarked','Sex', 'PassengerId'], axis=1)	

	df = df.astype(float)
	print df.info()

	clean_data = df.values
#	print clean_data
	return clean_data


def clean_ages
