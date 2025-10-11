---
title: The Options and Knowledge (Oak) Architecture
date: 2025-08-1
is_post: true
---

I've recently been digging into Rich Sutton's current research direction, both from the [Alberta Plan](http://www.incompleteideas.net/Talks/AlbertaPlan.pdf) and recent talks he has been giving on aspects of the OaK architecture.

# Architecture

The OaK architecture aims to provide the most general description of a reinforcement learning agent that learns to
maximize its reward signal.

## Learn a Policy and Value function for Maximizing Rewards

This starts quite simple, we just perform normal RL to try to maximize our stream of rewards. Any off the shelf
algorithm will work, but the factor keeping this from being truly general is our lack of tools for nonlinear, continual
learning. Almost all Deep RL-based approaches require some way of turning RL problems into small batches of supervised
learning problems. The models we build don't actually engage with the nonstationarity of the real world and treat
everything as small datasets of stationary policy learning.

In order to truly conquer this step, we need methods that perform continual learning from a stream of interactions with
the environment. This means no replay buffers, feature representations that adapt to the changing distribution of the
world, and blah.

## Generalize New State Features from Existing Features

This is sort of linked to the continual learning problem, but as learning progress is made an agent needs to be able to
create or generalize state features from the ones they used to previously explore the environment. 
I am interested in why this is a separate step, as in standard Deep RL right now the feature learning/extraction happens
at the same time as policy and value function learning. End-to-End seems to be the winner all the time in deep learning,
but it seems he thinks as the perception stack as separate, or at least to think of it as separate.

## Rank by Attaining Rewards

This one is pretty straight forward, but it requires good value functions to tell you which features are most important
for getting high rewards. These are the features you want to be able to recreate easily in your environment. This is key
for getting around the large world hypothesis, as you only focus on a handful of things to improve upon at a time.


## Generate Subproblems for highly Ranked Features

What are good subproblems to focus on?

"Reward Respecting subproblems of Feature Attainment"

Reward Respecting - Do these features lead extremes of cumulative reward?
Feature Attainment - Can you easily reach these features?


## Learn Solutions to Subproblems

Solutions to subproblems are essentially "Options", or temporally extended action sequences that have an associated
termination condition. Options provide a general way to learn reusable solutions to these feature attainment problems.

Option-based RL

## Learn Transition models from those Solutions

Once we learn new options, we need to update our transition model of the world to understand how our new options are
able to move us from certain initial features to final ones. 

Option-based Transition Models

## Plan with learned Transition Models

Approximate planning w/ state features and learned transition models.

## Keep Statistics on quality of Model and Features

This allows the agent to re-evaluate the utility of learned models, features, options. 
