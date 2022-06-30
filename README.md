# neural-network-autotrain-digital-counter
Action and script controlled automated training of neural network to readout the value of a digital counter. This is an via GitHub Actions automatized version of https://github.com/jomjol/neural-network-digital-counter-readout.

There you also find further background information.

### Working principle:

1) New training image of a digital counter is uploaded to the directory `/ziffer_raw/` is uploaded
2) GitHub action for a training cycle including the new images is started
   * starting point: old trained neural  network (loading of the old training status)
   * different versions of the neural network are trained in a dedicated docker container
3) Standardized report are generated to evaluate the training success
4) Reports and results are pushed to the GitHub for further investigations and usage
   * Old training status is stored additionally (reports and neural network configurations)



The neural network is used within the "AI-on-the-edge" project to digitize different analog meters (watermeter, gasmeter, ...) as the a water meter measurement system. An overview can be found here: [https://github.com/jomjol/AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device)

#### 1.2.2 New Images - (2022-06-30)

* Accumulate new images

#### 1.5.0 New Images - (2022-04-24)

* New Images

#### 1.4.0 New Images - (2022-03-15)

* Techem images

#### 1.3.1 New Images - (2022-02-27)

* new images

#### 1.3.0 New Images - (2022-02-23)

* new images series (ISKRA meters)

#### 1.2.3 New Images - (2022-02-23)

* new images

#### 1.2.2 New Images - (2022-02-11)

* new images

#### 1.2.1 New Images - (2022-02-05)

* new images

#### 1.2.0 Reset - (2022-01-21)

* Corrected wrong labeling (LCD) and reset the learing

#### 1.1.1 New Images - (2022-01-10)

* Updated LCD images

#### 1.1.0 New Images - (2022-01-02)

* Full set of LCD images (rather pasty)

#### 1.0.0 Initial Version - (2021-12-20)

* Initial Version


#### [Overview older Versions](Versions.md)





_______

## Description

Background and details for the neural network can be found in: https://github.com/jomjol/neural-network-digital-counter-readout

