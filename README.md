## Machine Learning  based stock trading system

### Overview

This is a machine learning technology based stock trading system, for modelling stock trading of AAPL.  In the system, there are three different trading strategies: (1) manual rule based strategy, (2) ML decision tree based strategy, (3) Q-Learner based strategy. Along with the python code for each strategy, there are also visualiztion and comparative analysis. You will see a well formatted report (report.pdf) for the detailed discussion. Although the system is tested with only AAPL, the knowledge and technique can be extended to any stock with modification.

### Manual rule based trading strategy

In this part, first I will develop and descripe 3 different popular technical indications: Price/SMA, Stochastic Oscillator %D and Bollinger Band %B indicators. In the report part 1, you can see the details about the analysis and visualization of these indicators. The corresponding python code is in indicators.py file.  After that we need to design a manual rule based trading system, which is like a traditional trading strategy. The result and performance is well descriped in part 3 of the report. The related python code is in  rule_based.py.

### ML decision tree based strategy

In the part, we will need to use machine learning knowledge to build a ML based trading strategy. We will first develop a random decision tree learner and a bagger learner. After that we will construct the ML based strategy and compare it with rule based one. The performance and analysis is discussed in part 4 of the report. The corresponding code is RTLearner.py, BagLearner.py and ML_based.py.  (note that there are some utility functions that will be used by these files, I put them in analysis.py, marketsim.py and util.py).

### Q-Learner based strategy

In this part, we will develop a Q-learner and apply it to the trading problem. We will use the indicators we developed before as state, and for actions we will use BUY, NOTHING, and SELL. The corresponding python code is in QLearner.py and StrategyLearner.py.






