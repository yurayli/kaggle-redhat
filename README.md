## Code for Red Hat Kaggle competition

Competition page
https://www.kaggle.com/c/predicting-red-hat-business-value


features:
almost all categorical features, and some with high cardinality;
one-hot encoding for categorical features; discard 'char_10' in training data;
cross validation set split by 'people_id'

#### xgboost gbtree
model: performance better without one-hot encoding.

<th>Performance:</th>
<table>
    <tr><td></td><td>CV score</td><td>Public</td><td>Private</td></tr>
    <tr><td>No leak</td><td>0.947468</td><td>0.953907</td><td>0.953896</td></tr>
    <tr><td>With leak</td><td>N/A</td><td>0.990610</td><td>0.990595</td></tr>
</table>

#### xgboost gblinear
model: using sparse data.

<th>Performance:</th>
<table>
    <tr><td></td><td>CV score</td><td>Public</td><td>Private</td></tr>
    <tr><td>No leak</td><td>0.979611</td><td>0.980765</td><td>0.980584</td></tr>
    <tr><td>With leak</td><td>N/A</td><td>0.990158</td><td>0.990171</td></tr>
</table>

#### neural net
model: with embedding layer on 'group_1' + Batch Normalization + Dropout

<th>Performance:</th>
<table>
    <tr><td></td><td>CV score</td><td>Public</td><td>Private</td></tr>
    <tr><td>No leak</td><td>0.985189</td><td>0.988611</td><td>0.988523</td></tr>
    <tr><td>With leak</td><td>N/A</td><td>0.990979</td><td>0.990986</td></tr>
</table>

#### ensembling
average of 6 nn + gbl + 3 gbt
<th>Best performance: (~22% of leaderboard)</th>
<table>
    <tr><td></td><td>Public</td><td>Private</td></tr>
    <tr><td>No leak</td><td>0.987725</td><td>0.987664</td></tr>
    <tr><td>With leak</td><td>0.991087</td><td>0.991075</td></tr>
</table>