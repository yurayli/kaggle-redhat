## Code for Red Hat Kaggle competition

Competition page  
https://www.kaggle.com/c/predicting-red-hat-business-value  

#### xgboost
features: discard 'char_10' in training data; onehot-encoding categorical features  
model: using gblinear booster and not too much regularization.  

<th>Best performance:</th>
<table>
    <tr><td>Private</td><td>Public</td></tr>
    <tr><td>0.980734</td><td>0.980902</td></tr>
</table>

#### neural net
features: same as above  
model: with embedding layer on 'group_1' + Batch Normalization + Dropout

<th>Best performance:</th>
<table>
    <tr><td>Private</td><td>Public</td></tr>
    <tr><td>0.987821</td><td>0.987988</td></tr>
</table>

#### ensembling
average of 6 DNN models
<th>Best performance: (~30% of leaderboard)</th>
<table>
    <tr><td>Private</td><td>Public</td></tr>
    <tr><td>0.989457</td><td>0.989581</td></tr>
</table>