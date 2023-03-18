# Anomaly Detection of Car Sensor Readings 

This involves the use of AutoEncoders to detect anomalies in sensor readings that could be gotten from a Car's CAN bus.


# Intuition
Training an autoencoder forces the model to learn useful representations that allow it to reconstruct the input with a lower dimensional representation.

The idea is if some arbitrary input does not follow the distribution that was learned by the autoencoder, the output of the autoencoder should be farther away from the input than normal.

This could be caused by:

1. The autoencoder *hallucinating* how the input should be, hence trying to "correct" the input.
2. The autoencoder making unreasonable predictions.

# Implementation
[The data was obtained from here](https://zenodo.org/record/3267184/files)

Two models were trained
  
- Densely connected Autoencoder
- Sequential Autoencoder / RNN Autoencoder

In the training of the first model, each timestep was considered independent and the input to the model were the sensor readings at that timestep.

After experiments, I concluded it would be better to include temporal context because realistically, the reading at a particular timestep is a function of the previous timesteps.