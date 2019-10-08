# Neural Network
Project for "Sistemas de inteligencia artificial" @ITBA, written in Octave

## Objective
Implement a neural network in octave. Train it for aproximating a terrain: given some coordenates we want to 
train the net to give us the altitude of such coordinates.

## Requisites

* Octave

## Running the net

Open octave and simply run
``` 
octave:1> runMulticapa
```


## Arguments



## Saving a state
When finished the learn algorithm all the data will be in global variable `P`
for saving it to a file for later analysis you can execute: `save('filename','P')`

For example:
```
octave:1> save('test1.m','P');
```

## Loading a state
For loading a previous saved state you will need to execute: `load('filename')`

For example:
```
octave:1> load('test1.m');
```
