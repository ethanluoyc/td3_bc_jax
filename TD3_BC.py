from typing import NamedTuple, Any
import copy
import numpy as onp
import haiku as hk
import jax.numpy as jnp
import jax
import optax
import pickle


def mse_loss(a, b):
    return jnp.mean(jnp.square(a - b))


class TrainingState(NamedTuple):
    actor_params: Any
    critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    actor_target_params: Any
    critic_target_params: Any


class Actor(hk.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = hk.Linear(256)
        self.l2 = hk.Linear(256)
        self.l3 = hk.Linear(action_dim)

        self.max_action = max_action

    def __call__(self, state):
        a = jax.nn.relu(self.l1(state))
        a = jax.nn.relu(self.l2(a))
        return self.max_action * jnp.tanh(self.l3(a))


class Critic(hk.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = hk.Linear(256)
        self.l2 = hk.Linear(256)
        self.l3 = hk.Linear(1)

        # Q2 architecture
        self.l4 = hk.Linear(256)
        self.l5 = hk.Linear(256)
        self.l6 = hk.Linear(1)

    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=1)

        q1 = jax.nn.relu(self.l1(sa))
        q1 = jax.nn.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = jax.nn.relu(self.l4(sa))
        q2 = jax.nn.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = jnp.concatenate([state, action], 1)

        q1 = jax.nn.relu(self.l1(sa))
        q1 = jax.nn.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):

        actor_transformed = hk.without_apply_rng(
            hk.transform(lambda obs: Actor(state_dim, action_dim, max_action)(obs))
        )
        critic_transformed = hk.without_apply_rng(
            hk.transform(lambda obs, a: Critic(state_dim, action_dim)(obs, a))
        )

        self.actor_optimizer = optax.adam(3e-4)
        self.critic_optimizer = optax.adam(3e-4)
        self._actor_apply = actor_transformed.apply
        self._critic_apply = critic_transformed.apply

        self._rng = hk.PRNGSequence(0)

        def init_state():
            actor_params = actor_transformed.init(
                next(self._rng), jnp.zeros((1, state_dim))
            )
            actor_opt_state = self.actor_optimizer.init(actor_params)
            critic_params = critic_transformed.init(
                next(self._rng), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim))
            )
            critic_opt_state = self.actor_optimizer.init(critic_params)
            return TrainingState(
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                copy.deepcopy(actor_params),
                copy.deepcopy(critic_params),
            )

        self._state = init_state()

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

        @jax.jit
        def _update_actor(actor_params, opt_state, key, critic_params, state, action):
            def loss_fn(actor_params):
                pi = self._actor_apply(actor_params, state)
                Q, _ = self._critic_apply(critic_params, state, pi)
                lmbda = jax.lax.stop_gradient(self.alpha / jnp.abs(Q).mean())
                # actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
                actor_loss = -lmbda * Q.mean() + jnp.mean(jnp.square(pi - action))
                return actor_loss

            loss, grad = jax.value_and_grad(loss_fn)(actor_params)
            update, opt_state = self.actor_optimizer.update(grad, opt_state)
            params_actor = optax.apply_updates(actor_params, update)
            return params_actor, opt_state, loss

        @jax.jit
        def _update_critic(
            params_critic,
            opt_state,
            key,
            critic_target_params,
            actor_params,
            actor_target_params,
            state,
            action,
            reward,
            next_state,
            not_done,
        ):
            def loss_fn(critic_params):
                # Select action according to policy and add clipped noise
                noise = jnp.clip(
                    jax.random.normal(key, action.shape),
                    -self.noise_clip,
                    self.noise_clip,
                )

                next_action = jnp.clip(
                    self._actor_apply(actor_target_params, next_state) + noise,
                    -self.max_action,
                    self.max_action,
                )

                # Compute the target Q value
                target_Q1, target_Q2 = self._critic_apply(
                    critic_target_params, next_state, next_action
                )
                target_Q = jnp.minimum(target_Q1, target_Q2)
                target_Q = jax.lax.stop_gradient(
                    reward + not_done * self.discount * target_Q
                )

                # Get current Q estimates
                current_Q1, current_Q2 = self._critic_apply(
                    critic_params, state, action
                )

                critic_loss = mse_loss(current_Q1, target_Q) + mse_loss(
                    current_Q2, target_Q
                )
                return critic_loss

            loss, grad = jax.value_and_grad(loss_fn)(params_critic)
            update, opt_state = self.critic_optimizer.update(grad, opt_state)
            params_critic = optax.apply_updates(params_critic, update)
            return params_critic, opt_state, loss

        self._update_actor = _update_actor
        self._update_critic = _update_critic

    def select_action(self, state):
        return onp.squeeze(
            onp.asarray(
                self._actor_apply(self._state.actor_params, state.reshape((1, -1)))
            ),
            axis=0,
        )

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        critic_params, critic_opt_state, _ = self._update_critic(
            self._state.critic_params,
            self._state.critic_opt_state,
            next(self._rng),
            self._state.critic_target_params,
            self._state.actor_params,
            self._state.actor_target_params,
            state,
            action,
            reward,
            next_state,
            not_done,
        )
        self._state = self._state._replace(
            critic_params=critic_params, critic_opt_state=critic_opt_state
        )

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            actor_params, actor_opt_state, _ = self._update_actor(
                self._state.actor_params,
                self._state.actor_opt_state,
                next(self._rng),
                self._state.critic_params,
                state,
                action,
            )
            self._state = self._state._replace(
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
            )
            # Update frozen target models
            self._state = self._state._replace(
                actor_target_params=optax.incremental_update(
                    self._state.actor_params, self._state.actor_target_params, self.tau
                ),
                critic_target_params=optax.incremental_update(
                    self._state.critic_params,
                    self._state.critic_target_params,
                    self.tau,
                ),
            )

            # Update the frozen target models
            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            # 	target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            # 	target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self._state, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self._state = pickle.load(f)
