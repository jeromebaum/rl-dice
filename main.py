import tensorflow as tf
from tflib import complex_tuple


display = print

def main():
    simultaneous_games = 10
    memory_size = 20
    learnable_count = 2

    simultaneous_games = 2
    memory_size = 4
    learnable_count = 0

    assert learnable_count < memory_size
    assert memory_size > simultaneous_games
    
    initial_tokens = 2
    num_actions = 2  # number of different actions, 0..n-1

    def create_game(state, actions):
        """
        Create a net for a set of games.
        Inputs are the current state and actions to take.
        Outputs are the next state, reward, binary of whether the game is over,
        and a transition (this is just inputs + outputs).
        """
        dice, tokens = state
        with tf.name_scope("random"):
            random_dice = tf.random_uniform([simultaneous_games], 1, 6+1, tf.int32, name="dice")
        with tf.name_scope("defaults"):
            default_tokens = tf.add(tf.zeros([simultaneous_games], tf.int32), initial_tokens, name="default_tokens")
        with tf.name_scope("logic"):
            check_actions = tf.assert_greater_equal(tokens, actions, name="check_actions")
            with tf.control_dependencies([check_actions]):
                games_over = tf.cast(
                    tf.logical_or(
                        tf.equal(actions, 0),
                        tf.equal(tokens - actions, 0),
                    ),
                    tf.int32,
                    name="games_over"
                )
                rewards = tf.multiply(games_over, tf.cast(dice, tf.int32), name="rewards")
                next_tokens = tf.add(
                    (1-games_over) * (tokens - actions),
                    games_over * default_tokens,
                    name="next_tokens"
                )
                next_dice = tf.identity(random_dice, name="next_dice")
        coded_state = tf.transpose([dice, tokens], name="state")
        coded_next_state = tf.transpose([next_dice, next_tokens], name="next_state")
        # Run everything through a `tf.tuple` call as a kind of "join" operation.
        next_dice, next_tokens, rewards, games_over, *transitions = tf.tuple([
            next_dice, next_tokens, rewards, games_over,
            coded_state,
            actions, rewards, games_over,
            coded_next_state,
        ], name="transition")
        next_state = next_dice, next_tokens
        return next_state, rewards, games_over, transitions

    def create_q_model(state, parameters):
        """
        Create a model for Q(S,a), using the parameters passed in.
        
        Uses the first entry in the shape of state to determine the shape of the output tensor.
        This allows for running multiple models in parallel (off the same parameters).
        
        Returns an op node with the estimate of Q(S,a), for all a, using the model.
        """
        model_count = tf.cast(state.shape[0], tf.int32)
        estimates = tf.random_uniform([model_count, num_actions], 0, 10, tf.int32, name="estimates")
        return tf.cast(estimates, tf.float32)

    def create_parameters():
        """
        Create an object containing the parameters to pass into q_model.
        
        You can use this to create more than one copy of the parameters, if you need to.
        """
        ()

    def create_playing_agent(state, parameters):
        """
        Create an agent net that plays a set of games.
        
        Input is state, parameters.
        
        Output is actions.
        """
        playing_model = create_q_model(state, parameters)
        action_probabilities = tf.nn.softmax(playing_model, name="action_probabilities")
        actions = tf.reshape(tf.cast(tf.multinomial(action_probabilities, 1), tf.int32), [-1], name="actions")
        return actions

    def create_replay(memory, transitions, learnable_count):
        """
        Create an experience replay.
        
        Inputs: memory, new transitions to remember, number of transitions to output.
        
        Outputs: new memory; transitions from the replay, annotated with how long they've been in memory for.
        """
        def create_part_replay(memory, transitions, indices_to_keep, learnable_indices):
            """
            Create part of the replay, to handle a single variable (e.g. age, state, reward).
            
            Inputs: values in memory, values in transitions, indices to override, learnable indices.
            """
            memory_values_to_keep = tf.gather(
                memory,
                indices_to_keep,
                name="memory_values_to_keep")
            next_memory_values = tf.random_shuffle(
                tf.concat([transitions, memory_values_to_keep], 0),
                name="next_memory_values")
            learnable_values = tf.gather(
                next_memory_values,
                learnable_indices,
                name="learnable_values")
            return next_memory_values, learnable_values

        memory_ages, memory_transitions = memory
        nextgen_memory_ages = tf.add(memory_ages, 1, name="nextgen_memory_ages")
        transition_count = tf.cast(transitions[0].shape[0], tf.int32)
        transition_ages = tf.zeros([transition_count], tf.int32, name="transition_ages")
        memory_size = tf.cast(memory_ages.shape[0], tf.int32)
        indices_to_keep = tf.random_uniform(
            [memory_size - transition_count],
            0, memory_size,
            tf.int32,
            name="indices_to_replace")
        learnable_indices = tf.random_uniform(
            [learnable_count],
            0, memory_size,
            tf.int32,
            name="learnable_indices")
        [
            next_memory_ages, learnable_ages,
            next_memory_states, learnable_states,
            next_memory_actions, learnable_actions,
            next_memory_rewards, learnable_rewards,
            next_memory_games_over, learnable_games_over,
            next_memory_next_states, learnable_next_states,
        ] = tf.tuple(sum([
            list(create_part_replay(memory, transitions, indices_to_keep, learnable_indices))
            for memory, transitions
            in [(nextgen_memory_ages, transition_ages)] + list(zip(memory_transitions, transitions))
        ], []), name="replay_output")
        learnable_transitions = [
            learnable_states,
            learnable_actions, learnable_rewards, learnable_games_over,
            learnable_next_states,
        ]
        next_memory_transitions = [
            next_memory_states,
            next_memory_actions, next_memory_rewards, next_memory_games_over,
            next_memory_next_states,
        ]
        next_memory = next_memory_ages, next_memory_transitions
        learnables = learnable_ages, learnable_transitions
        return next_memory, learnables

    def create_memory(transitions_example, transition_count):
        """
        Create an object containing the memory for the experience replay.
        
        Inputs: example transitions object, count of transitions to hold.
        """
        def create_part_memory(variable_example):
            """
            Create part of the memory for the experience replay, to hold a single variable.
            
            Inputs: example tensor for the variable.
            """
            desired_shape = [transition_count] + variable_example.shape[1:].as_list()
            part_memory = tf.Variable(tf.zeros(desired_shape, variable_example.dtype), name="part_memory")
            return part_memory
        memory_ages, *memory_transitions = [
            create_part_memory(variable)
            for variable
            in [tf.zeros([transition_count], tf.int32)] + transitions_example
        ]
        memory = memory_ages, memory_transitions
        return memory

    with tf.name_scope("parameters"):
        parameters = create_parameters()

    with tf.name_scope("game"):
        with tf.name_scope("state"):
            dice = tf.Variable(tf.zeros([simultaneous_games], tf.int32), name="dice")
            tokens = tf.Variable(tf.ones([simultaneous_games], tf.int32), name="tokens")
            game_state = dice, tokens
        with tf.name_scope("input"):
            actions = tf.Variable(tf.zeros([simultaneous_games], tf.int32), name="actions")
        next_game_state, rewards, games_over, transitions = create_game(game_state, actions)
        next_dice, next_tokens = next_game_state

    with tf.name_scope("playing_agent"):
        coded_game_state = tf.transpose([dice, tokens], name="coded_game_state")
        agent_actions = create_playing_agent(coded_game_state, parameters)
        set_game_state = tf.group(
            tf.assign(dice, next_dice),
            tf.assign(tokens, next_tokens),
            name="set_game_state")
        set_actions = tf.assign(actions, agent_actions, name="set_actions")
        set_agent_next = tf.group(set_game_state, set_actions, name="set_agent_next")

    def zip_dict_with(fn, dictA, dictB):
        return dict([
            (key, fn(dictA[key], dictB[key]))
            for key in dictA.keys()
        ])

    def assign_tuple(refs, tensors, name=None):
        assigns = [
            tf.assign(ref, tensor)
            for ref, tensor
            in zip(refs, tensors)
        ]
        return tf.group(*assigns, name=name)

    def assign_dict(refs, tensors, name=None):
        keys = list(refs.keys())
        return assign_tuple(
            [refs[key] for key in keys],
            [tensors[key] for key in keys],
            name=name)

    with tf.name_scope("experience_replay"):
        memory = memory_ages, memory_transitions = create_memory(transitions, memory_size)
        next_memory, learnables = create_replay(memory, transitions, learnable_count)
        next_memory_ages, next_memory_transitions = next_memory
        set_memory = tf.group(
            tf.assign(memory_ages, next_memory_ages),
            assign_tuple(memory_transitions, next_memory_transitions),
            name="set_memory")

    set_next = tf.group(set_agent_next, set_memory, name="set_next")

    with tf.name_scope("hyperparameters"):
        ()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(set_next)
        display(sess.run({'transitions': transitions, 'memory': memory, 'set_next': set_next}))
        display(sess.run({'transitions': transitions, 'memory': memory, 'set_next': set_next}))
        display(sess.run({'transitions': transitions, 'memory': memory, 'set_next': set_next}))
        display(sess.run({'transitions': transitions, 'memory': memory, 'set_next': set_next}))

with tf.Graph().as_default():
    tf.set_random_seed(8440977762267892507)
    main()
