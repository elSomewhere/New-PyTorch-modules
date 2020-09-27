import torch
import torch.distributions.mixture_same_family
import torch.distributions.distribution
import torch.distributions.categorical
from torch.distributions import constraints
import numpy as np
from matplotlib import pylab as plt

# Implementation closely follows:
# https://github.com/tensorflow/probability/blob/v0.11.0/tensorflow_probability/python/distributions/hidden_markov_model.py


class HiddenMarkovModel(torch.distributions.Distribution):
    ''' Hidden Markov model distribution
    The `HiddenMarkovModel` distribution implements a (batch of) hidden
    Markov models where the initial states, transition probabilities
    and observed states are all given by user-provided distributions.
    This model assumes that the transition matrices are fixed over time.
    '''
    arg_constraints = {}
    has_rsample = False
    support = constraints.dependent

    def __init__(self,
                 initial_distribution,
                 transition_distribution,
                 observation_distribution,
                 num_steps,
                 validate_args=None):
        '''
        Construct a 'HiddenMarkovModel' distribution
        :param initial_distribution: A `Categorical`-like instance.
        Determines probability of first hidden state in Markov chain.
        The number of categories must match the number of categories of
        `transition_distribution` as well as both the rightmost batch
        dimension of `transition_distribution` and the rightmost batch
        dimension of `observation_distribution`.
        :param transition_distribution: A `Categorical`-like instance.
        The rightmost batch dimension indexes the probability distribution
        of each hidden state conditioned on the previous hidden state.
        :param observation_distribution: A `torch.distributions.distribution.Distribution`-like
        instance.  The rightmost batch dimension indexes the distribution
        of each observation conditioned on the corresponding hidden state.
        :param num_steps: The number of steps taken in Markov chain. An integer valued
        tensor. The number of transitions is `num_steps - 1`.
        '''
        self._initial_distribution = initial_distribution
        self._transition_distribution = transition_distribution
        self._observation_distribution = observation_distribution
        self._num_steps = torch.tensor(num_steps)

        if not isinstance(self._initial_distribution, torch.distributions.Categorical):
            raise ValueError(" The initial distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        if not isinstance(self._transition_distribution, torch.distributions.Categorical):
            raise ValueError("The transition distribution need to be an "
                             "instance of torch.distributions.Categorical")

        if not isinstance(self._observation_distribution, torch.distributions.Distribution):
            raise ValueError("The observation distribution need to be an "
                             "instance of torch.distributions.Distribution")

        # infer event shape
        if num_steps is not None:
            if np.ndim(num_steps) != 0:
                raise ValueError(
                    '`num_steps` must be a scalar but it has rank {}'.format(
                        np.ndim(num_steps)))
            else:
                event_shape = torch.Size(torch.cat([torch.tensor([num_steps]),torch.tensor(self.observation_distribution.event_shape)]).type(torch.int))
        else:
            event_shape = torch.Size(torch.cat([torch.tensor([1]),torch.tensor(self.observation_distribution.event_shape)]).type(torch.int))

        # infer batch shape
        batch_shape = self._broadcast_shape(
            self.initial_distribution.batch_shape, self._broadcast_shape(self.transition_distribution.batch_shape[:-1], self.observation_distribution.batch_shape[:-1])
        )
        super(HiddenMarkovModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args
        )

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def initial_distribution(self):
        return self._initial_distribution

    @property
    def transition_distribution(self):
        return self._transition_distribution

    @property
    def observation_distribution(self):
        return self._observation_distribution

    # TODO: avoid the numpy operation, there may be some torhc function for this?
    def _broadcast_shape(self, torch_size_a, torch_size_b):
        res = torch.Size((torch.ones(torch_size_a) +  torch.ones(torch_size_b)).shape)
        return res

    def _batch_shape_tensor(self):
        res = self._broadcast_shape(
            self.initial_distribution.batch_shape,
            self._broadcast_shape(
                self.transition_distribution.batch_shape[:-1], self.observation_distribution.batch_shape[:-1]
            )
        )
        return res

    def _move_dimension(self, tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
        # see https://github.com/pytorch/pytorch/issues/36048
        dim = tensor.dim()
        perm = list(range(dim))
        if destination < 0:
            destination += dim
        perm.pop(source)
        perm.insert(destination, source)
        return tensor.permute(*perm)


    def _extract_log_probs(self, num_states, dist):
      states = torch.reshape(
          torch.arange(0, num_states),
          tuple(np.array(torch.cat([torch.tensor([num_states]),torch.ones_like(torch.tensor(dist.batch_shape))],axis=0).type(torch.int)))
      )
      log_prob = dist.log_prob(states)
      res = self._move_dimension(log_prob, 0, -1)
      return res

    def log_prob(self, value):
        observation_tensor_shape = value.shape
        observation_distribution = self.observation_distribution
        underlying_event_rank = len(observation_distribution.event_shape)
        observation_batch_shape = observation_tensor_shape[:-1 - underlying_event_rank]
        batch_shape = self._broadcast_shape(observation_batch_shape,self._batch_shape_tensor())
        num_states = self.transition_distribution.batch_shape[-1]
        log_init = self._extract_log_probs(num_states,self.initial_distribution)
        log_init = log_init.expand(torch.Size(tuple(torch.cat([torch.tensor(batch_shape), torch.tensor([num_states])], axis=0))))
        log_transition = self._extract_log_probs(num_states, self.transition_distribution)
        observation_event_shape = observation_tensor_shape[-1 - underlying_event_rank:]
        working_obs = value.expand(torch.Size(torch.cat([torch.tensor(batch_shape), torch.tensor(observation_event_shape)], axis=0)))
        r = underlying_event_rank
        working_obs = self._move_dimension(working_obs, -1 - r, 0)
        working_obs = torch.unsqueeze(working_obs, -1 - r)
        observation_probs = observation_distribution.log_prob(working_obs)
        def log_vector_matrix(vs, ms):
            return torch.logsumexp(vs.unsqueeze(-1) + ms, axis=-2)
        fwd_prob = log_init
        for i in range(observation_probs.shape[0]):
            fwd_prob = log_vector_matrix(fwd_prob, log_transition) + observation_probs[i,:,:]
        log_prob = torch.logsumexp(fwd_prob, axis=-1)
        res = log_prob
        return res

    def _observation_log_probs(self, observations, mask=None):
        observation_distribution = self.observation_distribution
        underlying_event_rank = len(observation_distribution.event_shape)
        observation_tensor_shape = observations.shape
        observation_batch_shape = observation_tensor_shape[:-1 - underlying_event_rank]
        observation_event_shape = observation_tensor_shape[-1 - underlying_event_rank:]
        if mask is not None:
            mask_tensor_shape = mask.shape
            mask_batch_shape = mask_tensor_shape[:-1]
        batch_shape = self._broadcast_shape(observation_batch_shape, self._batch_shape_tensor())
        if mask is not None:
            batch_shape = self._broadcast_shape(batch_shape, mask_batch_shape)
        observations = observations.expand(torch.Size(torch.cat([torch.tensor(batch_shape), torch.tensor(observation_event_shape)], axis=0)))
        observation_rank = len(observations.shape)
        observations = self._move_dimension(observations, observation_rank - underlying_event_rank - 1, 0)
        observations = torch.unsqueeze(observations, observation_rank - underlying_event_rank)
        _observation_log_probs = observation_distribution.log_prob(observations)
        return _observation_log_probs

    def _log_vector_matrix(self, vs, ms):
        return torch.logsumexp(vs.unsqueeze(2) + ms, axis=-2)

    def _log_matrix_vector(self, ms, vs):
        return torch.logsumexp(ms + vs.unsqueeze(len(vs.shape) - 1), axis=-1)

    def posterior_marginals(self, observations, mask=None):
        observation_tensor_shape = observations.shape
        observation_distribution = self.observation_distribution
        underlying_event_rank = len(observation_distribution.event_shape)
        observation_batch_shape = observation_tensor_shape[:-1 - underlying_event_rank]
        num_states = self.transition_distribution.batch_shape[-1]
        _observation_log_probs = self._observation_log_probs(observations, mask)
        log_init = self._extract_log_probs(num_states, self.initial_distribution)
        log_prob = log_init + _observation_log_probs[0]
        log_transition = self._extract_log_probs(num_states, self.transition_distribution)
        log_adjoint_prob = torch.zeros_like(log_prob)

        def log_vector_matrix(vs, ms):
            return torch.logsumexp(vs.unsqueeze(len(observation_batch_shape)) + ms, axis=-2)

        def forward_step(log_previous_step, log_prob_observation):
            return log_vector_matrix(log_previous_step, log_transition) + log_prob_observation

        temp = forward_step(log_prob, _observation_log_probs[1, :, :])
        forward_log_probs = temp.unsqueeze(0)
        for i in range(2, _observation_log_probs.shape[0]):
            temp = forward_step(temp, _observation_log_probs[i, :, :])
            forward_log_probs = torch.cat((forward_log_probs, temp.unsqueeze(0)), axis=0)
        forward_log_probs = torch.cat([log_prob.unsqueeze(0), forward_log_probs], axis=0)
        total_log_prob = torch.logsumexp(forward_log_probs[-1], axis=-1)

        def backward_step(log_previous_step, log_prob_observation):
            return self._log_matrix_vector(log_transition, log_prob_observation + log_previous_step)

        step = log_adjoint_prob
        backward_log_adjoint_probs = step.unsqueeze(0)
        for i in list(np.arange(1, _observation_log_probs.shape[0])[::-1]):
            step = backward_step(step, _observation_log_probs[i, :, :])
            backward_log_adjoint_probs = torch.cat([step.unsqueeze(0), backward_log_adjoint_probs], axis=0)
        log_likelihoods = forward_log_probs + backward_log_adjoint_probs
        marginal_log_probs = self._move_dimension(log_likelihoods - total_log_prob.unsqueeze(-1), 0, -2)
        res = torch.distributions.categorical.Categorical(logits=marginal_log_probs)
        return res

    def sample(self, sample_shape=torch.Size()):
        # given we set has_rsample = False, calling no_grad for consistency
        with torch.no_grad():
            n = torch.tensor(sample_shape)
            transition_batch_shape = self.transition_distribution.batch_shape
            num_states = torch.tensor(transition_batch_shape)[-1]
            batch_shape = self.batch_shape
            batch_size = torch.prod(torch.tensor(batch_shape))
            init_repeat = (torch.prod(torch.tensor(batch_shape)) // torch.prod(
                torch.tensor(self.initial_distribution.batch_shape))).type(torch.int)
            init_state = self.initial_distribution.sample(sample_shape=n * init_repeat)
            init_state = torch.reshape(init_state, torch.Size(torch.cat([n, batch_size.unsqueeze(0)]).type(torch.int)))
            transition_repeat = (
                        torch.prod(torch.tensor(batch_shape)) // torch.prod(torch.tensor(transition_batch_shape[:-1])))
            dummy_index = torch.zeros(self._num_steps - 1)
            state = init_state
            hidden_states = state.unsqueeze(0)
            for i in range(dummy_index.shape[0]):
                gen = self.transition_distribution.sample((n * transition_repeat).type(torch.int))
                new_states = gen.reshape(
                    torch.Size(torch.cat([n, batch_size.unsqueeze(0), num_states.unsqueeze(0)]).type(torch.int)))
                old_states_one_hot = torch.nn.functional.one_hot(state, num_states)
                result = torch.sum(old_states_one_hot * new_states, axis=-1)
                hidden_states = torch.cat([hidden_states, result.unsqueeze(0)])
                state = result
            hidden_one_hot = torch.nn.functional.one_hot(hidden_states, num_states)
            observation_repeat = (batch_size // torch.prod(torch.tensor(self.observation_distribution.batch_shape[:-1])))
            possible_observations = self.observation_distribution.sample(
                torch.Size(torch.cat([self._num_steps.unsqueeze(0), observation_repeat * n]).type(torch.int)))
            inner_shape = self.observation_distribution.event_shape
            possible_observations = possible_observations.reshape(
                torch.Size(
                    torch.cat(
                        [
                            torch.cat([self.num_steps.unsqueeze(0), n]),
                            torch.tensor(batch_shape),
                            torch.tensor([num_states]),
                            torch.tensor(inner_shape)
                        ]
                    ).type(torch.int)
                )
            )
            hidden_one_hot = hidden_one_hot.reshape(
                torch.Size(
                    torch.cat(
                        [
                            torch.cat([self.num_steps.unsqueeze(0), n]),
                            torch.tensor(batch_shape),
                            torch.tensor([num_states]),
                            torch.ones_like(torch.tensor(inner_shape))
                        ]
                    ).type(torch.int)
                )
            )
            observations = torch.sum(hidden_one_hot * possible_observations, axis=-1 - len(inner_shape))
            observations = self._move_dimension(observations, 0, len(batch_shape) + len(sample_shape))
            return observations

    #TODO
    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    #TODO
    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @constraints.dependent_property
    def support(self):
        #TODO: Supports are dependent on the distributions used to initialize. Possible to pass case-dependent supports?
        return self.support#self._observation_distribution


    #TODO
    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        raise NotImplementedError

if __name__ == "__main__":
    ''' 
    Example:
    Train a 2-state HMM with 3 discrete events using backpropagation
    '''

    # first, lets initialize some random HMM to create samples from

    # initial_distribution
    initial_distribution = torch.distributions.categorical.Categorical(torch.tensor([0.2, 0.8]))
    # transition_distribution
    transition_distribution = torch.distributions.categorical.Categorical(torch.tensor([[0.9, 0.1], [0.2, 0.8]]))
    # observation_distribution
    observation_distribution = torch.distributions.categorical.Categorical(torch.tensor([[0.5, 0.05, 0.45], [0.1, 0.8, 0.1]]))
    # length of the HMM sequence
    num_steps = 200
    # The actual HMM object
    sampling_hmm = HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=num_steps
    )
    #sample 300 instances of random series of length num_steps - we will train on these later
    y = sampling_hmm.sample(torch.Size((300,)))


    # Now lets create a new HMM object, this time with the goal of training it on above subseries.
    # Optimization of HMMs can be subject to degenerate minima if HMM weights are initialized to symmetric states.
    # Also, overall optimization may subject to many local minima.
    # We therefore train 30 parallel HMMs with random initialization,
    # hereby also taking advantage of pytorch's batch processing capabilities.
    # We will select the best model at the end of the training loop

    num_repeats = 30
    initial_logits = torch.randn([num_repeats, 1, 2], requires_grad=True)
    transition_logits = torch.randn([num_repeats, 1, 2, 2], requires_grad=True)
    observation_logits = torch.randn([num_repeats, 1, 2, 3], requires_grad=True)

    # define the train loop
    def train(y):
        trainable_hmm = HiddenMarkovModel(
            initial_distribution=torch.distributions.categorical.Categorical(logits=initial_logits),
            transition_distribution=torch.distributions.categorical.Categorical(logits=transition_logits),
            observation_distribution=torch.distributions.categorical.Categorical(logits=observation_logits),
            num_steps=num_steps)
        loss_all = trainable_hmm.log_prob(y)
        loss = -torch.mean(loss_all)
        return loss, loss_all,trainable_hmm

    # now lets train
    optimizer = torch.optim.Adam([initial_logits, transition_logits, observation_logits], lr=1e-1)
    epochs = 100

    loss_curve = []
    for epoch in range(epochs):
        loss, loss_all, current_model = train(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch: %s loss: %s' % (epoch, loss))
        loss_curve += [-loss_all.data.numpy().mean(axis=1)]
    loss_curve = np.array(loss_curve)
    plt.plot(loss_curve)
    best_model_idx = np.argmin(loss_curve[-1, :])
    print('Best model is at index %s' % best_model_idx)


    # Now lets check how well the model was trained
    print("initial_distribution")
    print(initial_distribution.probs)
    print(current_model.initial_distribution.probs[best_model_idx])
    print("transition_distribution")
    print(transition_distribution.probs)
    print(current_model.transition_distribution.probs[best_model_idx])
    print("observation_distribution")
    print(observation_distribution.probs)
    print(current_model.observation_distribution.probs[best_model_idx])