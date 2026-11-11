# Align4TSI

This is the code for paper:
> Align4TSI: Locality-Enhanced Cross-Variable Alignment Learning for Clinical Multivariate Time Series Imputation
> 


## Dependencies
* python==3.10
* numpy==1.24.4
* pandas==2.0.3
* scikit-learn==1.3.2
* scipy==1.10.1
* torch==2.0


## Datasets

The used datasets are available at:
* P12 https://physionet.org/content/challenge-2012/1.0.0/
* MIMIC https://physionet.org/content/mimiciii/1.4/
* P19 https://physionet.org/content/challenge-2019/1.0.0/




## Usage

```python
from align4tsi import Align4TSI, Align4TSIConfig
from align4tsi import mask_observations, masked_huber_loss


model = Align4TSI(Align4TSIConfig(json.load(stream)))
values, conditioning = mask_observations(truth, observation_mask, target_mask)
prediction = model(values, conditioning, target_mask,
                   valid_mask, scale_hours, variable_lengths)
loss = masked_huber_loss(prediction, truth, target_mask, valid_mask)

```


