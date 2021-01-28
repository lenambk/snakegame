#!/usr/bin/env python
# coding: utf-8

# In[1]:


#xay dung game
import pygame
import random
import time
import math
from tqdm import tqdm
import numpy as np



def display_snake(snake_position, display):
    for position in snake_position:
        pygame.draw.rect(display, (255, 0, 0), pygame.Rect(position[0], position[1], 10, 10))


def display_apple(apple_position, display):
    pygame.draw.rect(display, (0, 100, 0), pygame.Rect(apple_position[0], apple_position[1], 10, 10))


def starting_positions():
    snake_start = [100, 100]
    snake_position = [[100, 100], [90, 100], [80, 100]]
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score = 3

    return snake_start, snake_position, apple_position, score


def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position) - np.array(snake_position[0]))


def generate_snake(snake_start, snake_position, apple_position, button_direction, score):
    if button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10

    if snake_start == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0, list(snake_start))

    else:
        snake_position.insert(0, list(snake_start))
        snake_position.pop()

    return snake_position, apple_position, score


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_start):
    if snake_start[0] >= 500 or snake_start[0] < 0 or snake_start[1] >= 500 or snake_start[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_start, snake_position):
    if snake_start in snake_position[1:]:
        return 1
    else:
        return 0


def blocked_directions(snake_position):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])

    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    is_front_blocked = is_direction_blocked(snake_position, current_direction_vector)
    is_left_blocked = is_direction_blocked(snake_position, left_direction_vector)
    is_right_blocked = is_direction_blocked(snake_position, right_direction_vector)

    return current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked


def is_direction_blocked(snake_position, current_direction_vector):
    next_step = snake_position[0] + current_direction_vector
    snake_start = snake_position[0]
    if collision_with_boundaries(next_step) == 1 or collision_with_self(next_step.tolist(), snake_position) == 1:
        return 1
    else:
        return 0


def generate_random_direction(snake_position, angle_with_apple):
    direction = 0
    if angle_with_apple > 0:
        direction = 1
    elif angle_with_apple < 0:
        direction = -1
    else:
        direction = 0

    return direction_vector(snake_position, angle_with_apple, direction)


def direction_vector(snake_position, angle_with_apple, direction):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    new_direction = current_direction_vector

    if direction == -1:
        new_direction = left_direction_vector
    if direction == 1:
        new_direction = right_direction_vector

    button_direction = generate_button_direction(new_direction)

    return direction, button_direction


def generate_button_direction(new_direction):
    button_direction = 0
    if new_direction.tolist() == [10, 0]:
        button_direction = 1
    elif new_direction.tolist() == [-10, 0]:
        button_direction = 0
    elif new_direction.tolist() == [0, 10]:
        button_direction = 2
    else:
        button_direction = 3

    return button_direction


def angle_with_apple(snake_position, apple_position):
    apple_direction_vector = np.array(apple_position) - np.array(snake_position[0])
    snake_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])

    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10

    apple_direction_vector_normalized = apple_direction_vector/norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector/norm_of_snake_direction_vector
    angle = math.atan2(
        apple_direction_vector_normalized[1]*snake_direction_vector_normalized[0]-apple_direction_vector_normalized[0]*snake_direction_vector_normalized[1],
        apple_direction_vector_normalized[1]*snake_direction_vector_normalized[1]+apple_direction_vector_normalized[0]*snake_direction_vector_normalized[0])/math.pi
    return angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized


def play_game(snake_start, snake_position, apple_position, button_direction, score, display, clock):
    crashed = False
    while crashed is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        display.fill((20, 180, 255))

        display_apple(apple_position, display)
        display_snake(snake_position, display)

        snake_position, apple_position, score = generate_snake(snake_start, snake_position, apple_position,
                                                               button_direction, score)
        pygame.display.set_caption("SCORE: " + str(score))
        pygame.display.update()
        clock.tick(50000)

        return snake_position, apple_position, score


# In[2]:


#tao du lieu
from game import *

def generate_training_data(display, clock):
    training_data_x = []
    training_data_y = []
    training_games = 1000
    steps_per_game = 2000

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score = starting_positions()
        prev_apple_distance = apple_distance_from_snake(apple_position, snake_position)

        for _ in range(steps_per_game):
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            direction, button_direction = generate_random_direction(snake_position, angle)
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)

            direction, button_direction, training_data_y = generate_training_data_y(snake_position, angle_with_apple,
                                                                                    button_direction, direction,
                                                                                    training_data_y, is_front_blocked,
                                                                                    is_left_blocked, is_right_blocked)

            if is_front_blocked == 1 and is_left_blocked == 1 and is_right_blocked == 1:
                break

            training_data_x.append(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0], 
                 snake_direction_vector_normalized[0], apple_direction_vector_normalized[1], 
                 snake_direction_vector_normalized[1]])

            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)

    return training_data_x, training_data_y


def generate_training_data_y(snake_position, angle_with_apple, button_direction, direction, training_data_y,
                             is_front_blocked, is_left_blocked, is_right_blocked):
    if direction == -1:
        if is_left_blocked == 1:
            if is_front_blocked == 1 and is_right_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
                training_data_y.append([0, 0, 1])#re phai
            elif is_front_blocked == 0 and is_right_blocked == 1:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 0)
                training_data_y.append([0, 1, 0])#di thang
            elif is_front_blocked == 0 and is_right_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
                training_data_y.append([0, 0, 1])#re phai

        else:
            training_data_y.append([1, 0, 0])#re trai

    elif direction == 0:
        if is_front_blocked == 1:
            if is_left_blocked == 1 and is_right_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
                training_data_y.append([0, 0, 1])#re phai
            elif is_left_blocked == 0 and is_right_blocked == 1:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, -1)
                training_data_y.append([1, 0, 0])#re trai
            elif is_left_blocked == 0 and is_right_blocked == 0:
                training_data_y.append([0, 0, 1])#re phai
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
        else:
            training_data_y.append([0, 1, 0])#di thang
    else:
        if is_right_blocked == 1:
            if is_left_blocked == 1 and is_front_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 0)
                training_data_y.append([0, 1, 0])#di thang
            elif is_left_blocked == 0 and is_front_blocked == 1:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, -1)
                training_data_y.append([1, 0, 0])#re trai
            elif is_left_blocked == 0 and is_front_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, -1)
                training_data_y.append([1, 0, 0])#re trai
        else:
            training_data_y.append([0, 0, 1])#re phai

    return direction, button_direction, training_data_y


# In[3]:


from game import *

display_width = 500
display_height = 500
red = (255,0,0)
black = (0,0,0)
blue = (20,180,255)

pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()

training_data_x, training_data_y = generate_training_data(display,clock)


# In[4]:


# xay dung module
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


# In[5]:


model = Sequential()
model.add(Dense(units=9,input_dim=7, activation='relu'))

model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=30, activation='relu'))


model.add(Dense(3,  activation = 'softmax'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
history=model.fit(np.array(training_data_x).reshape(-1, 7),
                  np.array(training_data_y).reshape(-1, 3), 
                  batch_size = 256,
                  epochs= 10,
                  validation_split=0.25,
                  verbose=1)
# luu weight da training vao file 'weight.h5'
model.save_weights('weight.h5')
model_json = model.to_json()
with open('weight.json', 'w') as json_file:
    json_file.write(model_json)


# In[6]:


#ve bieu do accuracy và loss
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[7]:


#Test
from game import *
from keras.models import model_from_json

def run_game_with_ML(model, display, clock):
    max_score = 3
    avg_score = 0
    test_games = 100
    steps_per_game = 2000

    for _ in range(test_games):
        snake_start, snake_position, apple_position, score = starting_positions()

        count_same_direction = 0
        prev_direction = 0

        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            predictions = []

            predicted_direction = np.argmax(np.array(model.predict(np.array([is_left_blocked, is_front_blocked,                                                                              is_right_blocked,
                                                                             apple_direction_vector_normalized[0], \
                                                                             snake_direction_vector_normalized[0],
                                                                             apple_direction_vector_normalized[1], \
                                                                             snake_direction_vector_normalized[
                                                                                 1]]).reshape(-1, 7)))) - 1

            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)

            next_step = snake_position[0] + current_direction_vector
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                break
            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)

            if score > max_score:
                max_score = score

        avg_score += score

    return max_score, avg_score/100


json_file = open('weight.json', 'r')
loaded_json_model = json_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('weight.h5')


display_width = 500
display_height = 500
pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()
max_score, avg_score = run_game_with_ML(model,display,clock)
print("Maximum score achieved is:  ", max_score)
print("Average score achieved is:  ", avg_score)


# In[6]:





# In[ ]:





# In[37]:





# In[ ]:





# In[43]:





# In[ ]:





# In[ ]:




