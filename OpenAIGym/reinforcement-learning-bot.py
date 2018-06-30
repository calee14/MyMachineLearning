'''
Reinforcement Learning with OpenAI Gym
---
This notebook will create and test different reinforcement learning agents and environments.

'''
# Reinforcement Learning refers to problems where an agent needs to select actions in an evironment, and those actions change the state of the environment, resulting in rewards or loss of the game
# Open AI Gym is a python environment that provides an interface for creating reinforcement learning agents. 
# Each gum environment accepts actions each step, provides observations representing the current state of the environment, and provides rewards representing the agent's performance in the game so far
# Action - the action the bot should make
# Observation - the pixels on screen
# Rewards - the players score so far
import tensorflow as tf
import gym

import os

import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

'''
Load the Environment
---
Call `gym.make("environment name")` to load a new environment.

Check out the list of available environments at <https://gym.openai.com/envs/>

Edit this cell to load different environments!

'''
# TODO: Load an environment
# Gym loads a learning environment based on the id passed to it
env = gym.make("CartPole-v1")

# TODO: Print observation and action spaces
# Each environment has observations and possible actions that the agent will see and choose from. 
# Knowing the possible values that the enviroment will be passing you is key to processing your data and building your agents
print(env.observation_space)
print(env.action_space)

'''
Run an Agent
---

Reset the environment before each run with `env.reset`

Step forward through the environment to get new observations and rewards over time with `env.step`

`env.step` takes a parameter for the action to take on this step and returns the following:
- Observations for this step
- Rewards earned this step
- "Done", a boolean value indicating if the game is finished
- Info - some debug information that some environments provide. 

'''
# TODO Make a random agent
# Make an agent that plays the game with random actions to test out the environment
# Make a variable to set the max number of games to play
games_to_play = 10

# Create a for loop that runs that number of times of games to play
for i in range(games_to_play):
    # Reset the environment
    # Before each game you need to call env.reset() to initialize all variables and prepare the game to run
    # This returns the observations for the first frame of the game, which is saved as obs
    obs = env.reset()
    # Create a variable for the episode's rewards
    episode_rewards = 0
    # Create a boolean named done
    # To keep track of the bot to see if it's still playing the emulator
    done = False
    
    # Each step of the game render the environment, choose a random action, then take step forward with the chosen action
    # Create a while loop that runs as long as the done variable is false
    while not done:
        # Render the environment so we can watch
        env.render()
         
        # Choose a random action
        action = env.action_space.sample()
         
        # Take a step in the environment with the chosen action
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
 
    # Print episode total rewards when done
    print(episode_rewards)
     
# Close the environment
env.close()

'''
Policy Gradients
---
The policy gradients algorithm records gameplay over a training period, then runs the results of the actions chosen through a neural network, making successful actions that resulted in a reward more likely, and unsuccessful actions less likely.

'''
# TODO Build the policy gradient neural network

# In a reinforcement learning environment, your only indicator of the agent's success is the rewards, which can be sporadic. 
# On top of that, who's to say that the action the agent just took was actually responsible for the reward you received
# Since there isn't an easy way to use loss function, we'll need to come up with a new way to determine whether the actions our agent took were advantageous or not
# One method to estimate this is the policy gradients algorithm.
# The Policy Gradient algorithm, the agent plays the game and records each action, observation, and reward.
# Once the game is finished it tests which actions let to rewards, then adjusts the neural network to make those actions more likely and adjusts actions which led to lower rewards or loss of the game to be less likely

# The input data is the current state of the game, and the output is what action the agent should take
# Just like for the image classifier, you'll create a class that builds the neural network for your agent.

# Create a class called Agent
class Agent:
    
    # Create a constructor that takes two parameters - nums_actions and state_size
    # The constructor will let you adjust you model to work with different games that have different numbers of possible actions or different state data
    def __init__(self, num_actions, state_size):
        
        # Initializers can help start your network weights out with a bit of random value so they don't have to develop entirely from scratch.
        # One common initializer is xavier initializer
        # The zavier initializer initializes the starting values of the neurons
        initializer = tf.contrib.layers.xavier_initializer()
        
        # Create the input layer, a placeholder, to take in a number of states into the network in a batch
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        
        # Neural net starts here
        # The neural network will use dense of fully connected layers - every neuron is connected to each neuron in the previous layer
        
        # Use tf.layers.dense to create a hidden layer connected to the input layer with 8 units, relu activation, and the xavier initializer
        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        # Create a second layer connected to the first with the same units, activation, and initializer
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        
        
        # Output of neural net
        
        # Create a dense output layer connected to the second hidden layer with nun_actions units and no activation or initializer function
        out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)
        
        # Need the operations to get the networks estimated probabilities and the final actions choice for the input
        
        # The Softmax function will give you the softmaxed weights of each output action predicted by your network.
        # Because the outputs of softmax always add up to 1, you can use these as probabilities of a given action being taken.
        
        # Create a variable that runs softmax on the output layer
        self.outputs = tf.nn.softmax(out)
        # The Argmax function gets the index of the highest value in the input. You can use it to get the index of the chosen action
        # The axis=1 parameter indicates that you want the maximum value of axis 1 (the action weights)
        # With this, you can now get either action probabilities or an action choice out of the network
        # Create a variable that runs argmax on axis 1 of the output layer
        self.choice = tf.argmax(self.outputs, axis=1)
        
        # Training Procedure
        # In the training procedure the policy gradient network will be similar to the image recognition network, but with one complication
        # You don't know whether the actions the network chose were the correct move or not
        # You'll need to use the rewards gained during games to guide the learning process and decide which actions were more valuable
        # After each game, you'll pass the states observed, actions taken, and rewards earned into the network to decide how to update the network
        
        # Create two placeholders of shape [None,] for rewards and actions taken
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)
        
        # Use tf.one_hot to create a one hot encoding of actions taken
        one_hot_actions = tf.one_hot(self.actions, num_actions)
        
        # Calculate the cross entropy with the output layer and the one hot encodings.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)
        
        # Training procedure so far uses cross entropy just like the image classifier did, but you may have noticed that it only calculates the cross entropy between actions taken and the network's weights
        # This would be enough if you were already sure that every action taken was correct and should be learned from equally, but for all we know every action in that episode was the wrong choice
        
        # This is where the rewards come in. You can use the rewards earned in the episode to quide the learning towards more successful actions
        # Add a variable self.loss that multiplies the cross entropy by the tensor of rewards earned, then uses get the mean of the result
        self.loss = tf.reduce_mean(cross_entropy * self.rewards)
        
        # Calculate the gradients that would update the network based on this loss result
        # Create a variable that calls tf.gradients
        self.gradients = tf.gradients(self.loss, tf.trainable_variables())
         
        # In this model calculate gradients before applying them, which allows you to store up a few gradients before processing them in one big update
        # This prevents the network from leaning too much from a single game and allows you to adjust how often the network actually applies what it has learned and updates weights
        
        # Create a placeholder list for gradients
        
        # Create operations that will apply the gradients found
        
        # Create an empty list that will hold the placeholders for the gradients to apply
        self.gradients_to_apply = []
        # Create the for loop
        # The loop will run once for each trainable variable in your model
        for index, variable in enumerate(tf.trainable_variables()):
            # Create a tf.float placeholder
            gradient_placeholder = tf.placeholder(tf.float32)
            # Add the placeholder to the gradients list
            self.gradients_to_apply.append(gradient_placeholder)
         
        # Create the operation to update gradients with the gradients placeholder.
        optimizer = tf.train.AdamOptimizer(learning_rate=2e-2)
        # Update gradients operation applies the gradients fed in to their corresponding trainable variables in your model
        # Run the operation every time you want to actually apply what your model has learned from it's games and update its parameters
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))

'''
Discounting and Normalizing Rewards
---
In order to determine how "successful" a given action is, the policy gradient algorithm evaluates each action based on how many rewards were earned after it was performed in an episode.

The discount rewards function goes through each time step of an episode and tracks the total rewards earned from each step to the end of the episode.

For example, if an episode took 10 steps to finish, and the agent earns 1 point of reward every step, the rewards for each frame would be stored as 
`[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]`

This allows the agent to credit early actions that didn't lose the game with future success, and later actions (that likely resulted in the end of the game) to get less credit.

One disadvantage of arranging rewards like this is that early actions didn't necessarily directly contribute to later rewards, so a **discount factor** is applied that scales rewards down over time. A discount factor < 1 means that rewards earned closer to the current time step will be worth more than rewards earned later.

With our reward example above, if we applied a discount factor of .90, the rewards would be stored as
`[ 6.5132156   6.12579511  5.6953279   5.217031    4.68559     4.0951      3.439
  2.71        1.9         1. ]`

This means that the early actions still get more credit than later actions, but not the full value of the rewards for the entire episode.

Finally, the rewards are normalized to lower the variance between reward values in longer or shorter episodes.

You can tweak the discount factor as one of the hyperparameters of your model to find one that fits your task the best!
'''
# TODO Create the discounted and normalized rewards function

# To determine how successful a given action is, the policy gradient algorithm evaluates each action based on how many rewards were earned after it was performed in an episode
# Need a function that will go through each time step of an episode and track the total rewards earned from each step to the end of the episode
# Each step, a discount factor will scale rewards down over time so that actions that happen closer to the current time step are wighted higher than rewards earned later

# Create a variable that holds the discount rate
discount_rate = 0.9

# Declare the function that will track the rewards
def discount_normalize_rewards(rewards):
    # Create an empty numpy array of zeros the same size as the rewards list
    discounted_rewards = np.zeros_like(rewards)
    # Create a variable to track total rewards in th episode called total rewards set to 0
    total_rewards = 0
    
    # Now we need to go through each step and save the adjusted rewards value for each step
    # Create a for loop
    for i in reversed(range(len(rewards))):
        # Update total rewards
        total_rewards = total_rewards * discount_rate + rewards[i]
        # Save the current total reward to the discounted rewards list
        # This will go through and give each time step a weighted reward value based on how many rewards were earned from that step until the end of the episode
        discounted_rewards[i] = total_rewards
    
    # Normalizing the rewards helps keep reward values consistent across multiple game lengths
    # You wouldn't want your agent to skip out on learning from a valuable action just because the game it happened in was short
    # This process increases consistency and stability of reward values in the model
    
    # Subtract the mean of the discounted rewards list from each element of the rewards list
    discounted_rewards -= np.mean(discounted_rewards)
    # Divide each element by the standard deviation of the list after that
    discounted_rewards /= np.std(discounted_rewards)
    
    # Return the processed discounted rewards list
    return discounted_rewards

'''
Training Procedure
---
The agent will play games and record the history of the episode. At the end of every game, the episode's history will be processed to calculate the **gradients** that the model learned from that episode.

Every few games the calculated gradients will be applied, updating the model's parameters with the lessons from the games so far.

While training, you'll keep track of average scores and render the environment occasionally to see your model's progress.
'''
# TODO Create the training loop

# Reset the default graph
tf.reset_default_graph()

# Modify these to match shape of actions and states in your environment
# Initialize the variables with the number of actions and size of states for the environments
num_actions = 2
state_size = 4

# Make a new folder to hold checkpoints when using different environments
# Create a path for the saved checkpoints
path = "/home/student/Desktop/cartpole-pg/"

# Create variables to control how many episodes to train and training batch sizes
# These variables will control the training process
training_episodes = 1500
max_steps_per_episode = 10000
episode_batch_size = 10

# Create an agent using the class created
agent = Agent(num_actions, state_size)

# Create the global variables initializer
init = tf.global_variables_initializer()

# Initialize a saver to save model checkpoints
saver = tf.train.Saver(max_to_keep=2)

# If the path of the folder doesn't exist create one
if not os.path.exists(path):
    os.makedirs(path)

# Start a tensor flow session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    
    # Create an empty array to track the total rewards over the course of the session
    total_episode_rewards =  []
    
    # Create a buffer of 0'd gradients to store the calculated gradients while they wait to be applied
    # This line gets a gradient buffer that is the same size and shape of the model's currently trainable variables
    # Next you need to 0 out all the gradients so you can start fresh before learning
    gradient_buffer = sess.run(tf.trainable_variables())
    # Create a for loop
    for index, gradient in enumerate(gradient_buffer):
        # Multiply each gradient in the buffer by 0 to zero it out
        gradient_buffer[index] = gradient * 0
    
    # Each episode will represent one completed game in the environment.
    # The agent will loop through each episode, track its history, then move on to the next
    
    # Create a for loop that runs for the specified number of training episodes
    for episode in range(training_episodes):
        
        # At the start of each episode, reset the environment and save the resulting state
        state = env.reset()
        
        # The model will need to track the history of each episode (the states, actions, rewards and next state at each step of the episode)
        # This will allow it to process the data and calculate gradients
        
        # Create an empty list to hold the episode
        episode_history = []
        # Create a variable to track the total rewards for the current episode
        episode_rewards = 0
        
        # Create a for loop that runs until max steps per episode is reached
        for step in range(max_steps_per_episode):
             
            # Every 100 episodes render how the model would look
            if episode % 100 == 0:
                env.render()
             
            # Get weights for each action
            # This wil get the agents's guesses for which actions it thinks are more less or likely to succeed given the state passed in.
            # Next you'll convert that into an action choice you can pass to the gym
            action_probabilities = sess.run(agent.outputs, feed_dict={agent.input_layer: [state]})
            # Randomly choose out of the possible actions with weights from the softmax of the model
            action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])
            
            # Save the resulting states, rewards and whether the episode finished
            state_next, reward, done, _ = env.step(action_choice)
            # Append the state, action chosen, reward, and next state to the episode history
            episode_history.append([state, action_choice, reward, state_next])
            # Save the next sate as the current state
            state = state_next
            
            # Add rewards to the episode's reward total
            episode_rewards += reward
            
            # Check if the model finished an episode or ran out of steps
            if done or step + 1 == max_steps_per_episode:
                # Append the episodes rewards to the total rewards list
                total_episode_rewards.append(episode_rewards)
                # Convert the episode history to a numpy array
                episode_history = np.array(episode_history)
                # Run the discount and normalize rewards function on the stored rewards in episode history
                episode_history[:,2] = discount_normalize_rewards(episode_history[:,2])
                
                # Pass each column of the episode history into your model and run the gradients operation
                # The states column is stacked with np.vstack to match the list shape you specified when creating the input layer placeholder
                ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                                    agent.actions: episode_history[:,1],
                                                                    agent.rewards: episode_history[:, 2]})
                # add the gradients to the grad buffer:
                for index, gradient in enumerate(ep_gradients):
                    # Add the episode gradients to the corresponding index in the gradient buffer
                    gradient_buffer[index] += gradient
                # Break the for loop 
                break
        
        # Check if it's time to update the model with what it's learned
        if episode % episode_batch_size == 0:
            
            # Create a dict out of the gradient buffer with the following line
            feed_dict_gradients = dict(zip(agent.gradients_to_apply, gradient_buffer))
            
            # Update the model with the gradient dictionary
            sess.run(agent.update_gradients, feed_dict=feed_dict_gradients)
            
            # Zero out the gradients in the gradient buffer
            # Now the model can learn and improve
            for index, gradient in enumerate(gradient_buffer):
                gradient_buffer[index] = gradient * 0
            
        # Save checkpoint and print progress
        if episode % 100 == 0:
            saver.save(sess, path + "pg-checkpoint", episode)
            print("Average reward / 100 eps: " + str(np.mean(total_episode_rewards[-100:])))
                
'''
Testing the Model
---

This cell will run through games choosing actions without the learning process so you can see how your model has learned!
'''
# TODO Create the testing loop

# Create a variable to set how many episodes to use for testing
testing_episodes = 5
 
# Start a tensor flow session
with tf.Session() as sess:
    # Load the model's latest checkpoints
    # With the checkpoints you can test whether the agent has learned
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess,checkpoint.model_checkpoint_path)
    
    # Create a for loop that runs based on the number of testing episodes
    for episode in range(testing_episodes):
        
        # Reset the environment
        state = env.reset()
        
        # Create a variable called episode_reward to store this episode's rewards
        episode_rewards = 0
        
        # Create a for loop that runs through steps based on your max steps per episode
        for step in range(max_steps_per_episode):
            
            # Render the environment
            env.render()

            # Get Action
            # Pass the state into the model and run the choice operation
            action_argmax = sess.run(agent.choice, feed_dict={agent.input_layer: [state]})
            # Get the the agent's action choice based on the index from argmax
            action_choice = action_argmax[0]
            
            # Pass you choice to the environment the same way you did in the training loop
            state_next, reward, done, _ = env.step(action_choice)
            # Save the result next_state as state
            state = state_next
            
            # Add the reward from the last game step to the running total of episode rewards
            episode_rewards += reward
            
            # Create an if statement that checks if the episode is done or max steps have been reached
            if done or step + 1 == max_steps_per_episode:
                # Print the episode's rewards then break out of the for loop
                print("Rewards for episode " + str(episode) + ": " + str(episode_rewards))
                break

# Run to close the environment
env.close()