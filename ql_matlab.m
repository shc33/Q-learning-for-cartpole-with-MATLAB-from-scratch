% @author: Soohyun Cho
clear all; close all; format compact;

pe = pyenv();
env = py.gymnasium.make('CartPole-v1', pyargs('render_mode', "rgb_array"));

state_dim = int8(env.observation_space.shape(1)) % 4 (x, x', theta, theta')
action_dim = int8(env.action_space.n) % 2 (left, right)
disp("input dim: "+ state_dim +  ", output dim: "+ action_dim)
threshold = env.spec.reward_threshold;
disp("threshold: "+ threshold) % 475

obs_space_high = double(env.observation_space.high);
obs_space_low = double(env.observation_space.low);
state_bounds = [obs_space_low; obs_space_high]; 
disp("state_bounds" + mat2str(round(state_bounds,3)))

state_bounds(:, 2) = [-0.5; 0.5]; % change the bounds for x'
state_bounds(:, 4) = [-deg2rad(50); deg2rad(50)]; % change the bounds for theta'
disp("state_bounds" + mat2str(round(state_bounds,3)))

num_bins = [1, 1, 6, 3]; 
Q = zeros(num_bins(1), num_bins(2), num_bins(3), num_bins(4), action_dim);
disp("Q size: " + mat2str(size(Q)))

function bin_indexes = state_to_bin(state, state_bounds, num_bins)
    bin_indexes = [];
    for idx = 1:length(state)
        if state(idx) <= state_bounds(1, idx)
            bin_index = 1; % set it to min value.
        elseif state(idx) >= state_bounds(2, idx)
            bin_index = num_bins(idx);  % set it to max vaule.
        else % map state value to a bin
            bound_width = state_bounds(2, idx) - state_bounds(1, idx); 
            offset = (num_bins(idx)-1)*state_bounds(1, idx)/bound_width; 
            scaling = (num_bins(idx)-1)/bound_width;
            bin_index = round(scaling*state(idx) - offset); 
            bin_index = bin_index+1;
        end
        bin_indexes = [bin_indexes, bin_index];
    end
end

SEED = 1;
rng(SEED);
py.random.seed(int64(SEED));
py.numpy.random.seed(int64(SEED));
env.action_space.seed(int64(SEED));

min_learning_rate = 0.1;
learning_rate = 0.75;
discount_factor = 0.99;
epsilon = 1.0; % initial exploration rate
min_epsilon = 0.01; % minimum exploration rate
epsilon_decay = 0.99;
scores_array_max_length = 100;
scores_array = [];
max_step = 1000;
num_episodes = 1000;
for episode = 1:num_episodes
    obv = env.reset(pyargs('seed', int8(SEED)));
    obv = double(obv{1});
    state = int8(state_to_bin(obv, state_bounds, num_bins));
    step = 0;
    score = 0;
    while step < max_step
        step = step+1;
        temp = py.numpy.random.random();
        if temp < epsilon
            action_python = env.action_space.sample();
            action_python = int8(action_python);
            action = action_python + 1; % MATLAB's index starts from 1.
        else
            [~, action] = max(Q(state(1), state(2), state(3), state(4), :));
            action_python = int8(action - 1);
        end
        result = env.step(action_python);
        obv_prime = double(result{1});
        next_state = int8(state_to_bin(obv_prime, state_bounds, num_bins));
        reward = double(result{2});
        done = int8(result{3});
        score  = score + reward;
        if done == true
            reward = -200;
        end
        if 0
            img = double(env.render());
            img = uint8(img);
            imshow(img)
            hold on; axis on
            text(250, 50, "step="+step, 'Color', 'k', 'fontsize',15);
            pause(0.001);
        end
        best_q_val = max(Q(next_state(1), next_state(2), next_state(3), next_state(4), :));
        % Bellman equation
        Q(state(1), state(2), state(3), state(4), action) = Q(state(1), state(2), state(3), state(4), action) ...
            + learning_rate*(reward + discount_factor * best_q_val - Q(state(1), state(2), state(3), state(4), action));
        state = next_state;
        if done == true
            break;
        end
    end
    scores_array = [scores_array, score];
    if length(scores_array) > scores_array_max_length
        scores_array(1) = [];
    end
    avg_score = mean(scores_array);
    disp("Episode " + episode + " step " + step + ", eps = " + round(epsilon,4) ...
        + ", last " + length(scores_array) + " avg score = " + round(avg_score,4))
    if avg_score > threshold
        disp("Solved after " + episode + " episodes. Average Score:" + round(avg_score,4))
        break
    end
    if epsilon > min_epsilon
        epsilon = epsilon * epsilon_decay;
    else
        epsilon = min_epsilon;
    end
end
env.close();
