# Day-ahead electricity load forecasting

## Background

The electricity market is unique in the requirement that supply perfectly matches demand, since an economical use of current electricity storage technologies is simply not feasible, and an imbalance in supply and demand results in power system instability. This makes the price dynamics of the market extremely volatile and unlike any other, and means that accurate predictions for electricity loads (i.e. consumption) are extremely valuable for all market participants. For market-balancing authorities, estimations for how much electricity will need to be generated in the near future are essential for ensuring there is adequate and not excessive electricity generation scheduled in order to meet demand. For other market participants, volume forecasts are used for hedging risk and for their own production schedule.

## Project Overview

In this project, I employ time-series machine learning techniques with exogenous time features to forecast day-ahead electricity loads, using national grid data between 2011 and 2018.

To match the format of real-world short-term load forecasting, the predictions for electricity consumption in the 48 daily half-hour periods are dynamically established after the final period of the previous day. That is, the model is trained on the data up to and including today's load observations, and then tomorrow's forecasts are dynamically constructed.

This day-ahead dynamic forecasting method makes it unfeasible to use the SARIMAX module (which is very computationally expensive when dealing with large seasonal orders), and thus a large part of this project involved the use of object-oriented programming to create a class which performs the desired time-series forecasting. This class works with any inputted differencing steps and lags, such that we can find the predictions for any (differenced) AR process. I explain the processes within it in more detail in the Method section below.

## Data:

I used daily data from the national grid website from y/b 2011 to y/e 2018, found here https://www.nationalgrideso.com/data-explorer .  
Amongst other variables which I did not use for my analysis, the dataset comprised aggregate nation-wide electricity demand volumes, a variable indicating which of the 48 daily half-hour periods the observation relates to (period 1 being the half-hour beginning at 12.00am), and the date of the observation.

### Indexing error investigation

There was a slight issue in data as it was stored which required resolving. Upon inspection, I found that in each year, there was a day in March with load observations only up to period 46, and a day in October with 50 load observations.

It was important to determine whether this issue was a simple time-indexing error, where the load observations were in fact in the correct order, but the final two observations from the day in March were mistakenly entered as the first two observations of the following day, and then this shift lasted until the time-indices were 'corrected' in the day in October, where by entering 50 periods for that day, the time stamps are again correct the next day. Alternatively, the issue may have been only rooted in those two days. 

Through extensive visualisation of daily patterns before, on and after the anomolous days in several years, it was possible to identify convincing evidence that the period indices were shifted by the anoolies, in the sense that the daily patterns changed around them. Thus, the former issue was what was present in the data - an issue which was easy to solve, simply by dropping the period and date columns, and reconstructing them using the pandas daterange method.

## Method

I used object-oriented programming to create a class which could be used for constructing dynamic, day-ahead load forecasts for any specified autoregressive process, also allowing any specified differencing steps, as well as exogenous predictors. 

The steps behind these predictions were as follows:
### 1: Preprocessing

For the specified differencing steps and lags, the class has a method ( .preprocessing() ) which transforms the series into the differenced series, and then forms the target series and the predictor series (which is made up of the specified lags of the target series).

Exogenous regressors could also then be concatenated to this predictor set. These were simply the time-features for each observation: period (1-48), weekday (Mon-Sun), monthday (1-31), and month.

### 2: Dyamic predictions for the differenced series

Fitting the model on the data calibrates the lag coefficients, establishing the estimated dynamic process that the differenced data follows. Then, the next 48-periods are constructed according to this process, simply inputting the relevant lags. What makes this forecasting dynamic is that the final observation from the previous day (call it period 0), and so if the estimated dynamic process includes a lag from fewer than 48 periods ago, then at least some of these forecasts will rely on previous forecasts for their estimation.

### 3: 'Undifferencing' the forecasts

With these forecasts for the differenced series constructed, they require being translated back into an 'undifferenced' format - that is, into forecasts for actual electricity loads. In general, if you have a once-differenced series, undoing the differencing step requires adding back onto each observation its lag associated with the differencing step. I.e. if the series is first-differenced, you have to add back the first lag to each of the observations. This is done by placing the lag of the first observation at the beginning of the series, and using a cumulative sum. The cumulative sum is slightly more tricky for other differencing steps, but follows the same logic.

When it comes to undoing the differencing on the dynamic forecasts, things are a little more complicated, since the last actual electricity load we observe is period 0. So for most of these 48 periods to which we want to add back the differencing lag, that differencing lag hasn't actually happened yet, and so we need to use the corresponding forecast. Making the code which takes care of this for whatever combination of differencing steps you choose was a fun highlight of this project :) .

Once we have the dynamic day-ahead electricity load forecasts, we can compare them to the true loads using several metrics (I used root mean squared error, mean absolute error, and median absolute error).

# Results:

At this stage of the project, I have found that the best-fitting time-series process incorporates first differences, and uses lags 1 (the observation from half an hour ago), 48 (the same-period, previous-day observation), and 336 (same-period, same-weekday, previous-week). Forecasts are dynamically constructed for each 48-period day on the differenced series. Then, the forecasts required returning to the 'undifferenced' form, where they can be compared with actual demand observations to establish their accuracy, measured in terms of squared error. This 'undifferencing' required of course a custom function robust to the differencing steps and lags used. 

The best model also uses exogenous time features as predictors. These include period (1-48), weekday, monthday (1-31), and month. For this model, the coefficients are by far largest in absolute value for period dummy variables, i.e. for which period of the day the observation/forecast is in.

The best model achieved a mean squared error of 1929 (the series has a mean of 32,820, and a standard deviation of 7,478). While this is relatively accurate, there is significant scope for improvement. Firstly, it's well-established that electricity loads rely strongly on the weather (particularly temperature for heating/air conditioning considerations, but also windspeed and other factors). Incorporating weather forecast information is thus something which will likely yield significant improvement in accuracy, and which I plan to add to the project soon. Also, in much of the literature on electricity load forecasting, it's noted that the quality of forecasts is better when constructing separate time-series models for each period of the day. This is also something which I'll look into.

# Note on stationarity:

It was interesting in this project to delve into the issue of **stationarity** (where the statistical properties of the data are constant, i.e. don't depend on time). For electricity consumption data, stationarity does not hold, given the daily, weekly, monthly, and yearly recurring patterns.  

By visually inspecting autocorrelation and partial autocorrelation plots of the original series and of the series under differencing steps, as well as the form of the differenced series, I found that the data became stationary after differencing with steps (1, 1, 48, 336). However, while these differencing steps led to a strongly performing model in terms of predictions for the differenced series, the demand forecasts that resulted from dynamically reconstructing (dynamic, day-ahead) differenced forecasts were extremely poor.  

For all models tested, there was increasing error in forecasts as the period of the day increased, since errors accrue when they are dynamic. However, when the demand forecasts are constructed from predictions on a heavily differenced series, things get very inaccurate and volatile. This is because the heavily differenced series can be interpreted as something similar to the rate of change of the rate of change of the rate of change (etc.) between periods (which periods depends on the differencing steps specified). So, while predictions for this heavily differenced series may be very accurate, when the predictions are translated back into demand, there is much scope for wild runs. For example, if the forecasts for a double-differenced series (steps [1,1]) are mildly positive for five successive periods, then they can be interpreted as something like a sustained positive growth in the rate of increase between periods, which when translated back to the original series will look like an upwards curve which is increasingly steep.  

The aforementioned strongest model used a single differencing step. The electricity load data when differenced only once is not stationary (not even close). Yet, the demand predictions reconstructed from forecasts on this series are stronger than those when the series is differenced until stationary.


# Day-ahead-electricity-load-forecasting
