# Linear_Regression_Assignment

## Table of Contents

- General Info
- Key Objectives
- Dataset
- Technologies Used
- Conclusion


## General Info
A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. Company wants to understand the factors affecting the demand for these shared bikes in the American market.

## Key Objectives
Company wants to understand the factors affecting the demand for these shared bikes in the American market

- Which variables are significant in predicting the demand for shared bikes.
- How well those variables describe the bike demands.
## Business Goals
Build a model the demand for shared bikes with the available independent variables so that management wants

To understand how exactly the demands vary with different features.
Use the model to manipulate the business strategy to meet the demand levels and meet the customer's expectations.

## Dataset
day.csv from the Case Study

day.csv have the following fields:
	
	- instant: record index
	- dteday : date
	- season : season (1:spring, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2018, 1:2019)
	- mnth : month ( 1 to 12)
	- holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : temperature in Celsius
	- atemp: feeling temperature in Celsius
	- hum: humidity
	- windspeed: wind speed
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered

## Conclusions
Model Evaluation looks fine. Overall we have decent model

Features affecting demand are below :
- Temperature
- Season
- Working Day
- Weather Situation 
- Windspeed (inversely)


- Demand is high during Working days
- Demand is high during Summer seasons and less during Winter and Spring
- Demand is high when weather is Clear and less during Mist Cloudy and Light Snow
- Demand is high in 2019 compared to 2018
- Demand increases with temperature
- Demand decreases with windspeed

## Technologies Used
Python Libraries numpy, pandas, matplotlib.pyplot, seaborn, sklearn, statsmodel, linear regression
