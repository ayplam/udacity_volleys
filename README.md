### Project Details

For this project, a "Tennis" environment is used. In this environment, two agents need to learn how to hit a ball back and forth over the net. A reward of +0.1 is provided if the agent his the ball over net. A penalty of -0.01 is given if the ball hits the ground or if the agent hits the ball out of bounds. The goal for the agents are to keep hitting the ball back and forth for as long as possible.

The state space consists of 8 x 3 variables that correspond to how the agent sees the ball. Each action is a vector with 2 numbers, corresponding to moving forward or up (jumping) respectively. All actions should be clipped between [-1,1]. 

![Tennis](./imgs/tennis.png)


### Getting Started

##### Instructions

1. Environment Setup

In order to ensure that your environment has all the necessary dependencies, follow the steps [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)

2. Download the Environment

You will need to download the unity environment that matches your operating system below.

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Unzip the the downloaded environment in the same location as the ipynb.