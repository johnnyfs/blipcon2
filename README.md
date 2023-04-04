# blipcon

**Generative AI producing novel interactive classic-style games.**

This is a learning project aimed at exploring deep learning techniques for predicting video game states using transformer-like models. The objective is to create a model that can accurately predict game states based on current state and action inputs.

## Architectures Considered

There are three parts to the problem

* encode game states
* predict next state given previous state-action pair(s)
* embed the learning in a way that can be leveraged to generate novel sequences (ie, original games)

The architecture depends on the approach to the third, which is the least well-understood. I considered three approaches

* present the model with a VM & headless DSL; train it With RL to reproduce states it experiences by manipulating an AST according to rules in a "game" of game-making

* train the model as a hypernetwork to produce the weights for a "state machine" model whose scope is limited to mapping (s1, a) -> s2. Hypothetically, the weights (or specifically, the latents from which the weights would be decoded) would represent the "code" or "rules" of the state machine.

* build a transformer to predict states from sequences of state-action pairs

These approaches could be ranked in order of likely runtime performance from first to last, and in terms of ease of training/existing domain knowledge from last to first. (The two facts are related?) Further, the first would require a working state predictor to produce novel states to match, and the second would have to be trained by a state predictor capable of producing s2 to have any chance of success.

Given that, it made sense to attempt to build a state predictor first.

That problem presented two approaches to embedding states

* train the transformer to encode/decode the states it produces
* train a separate model to encode the states, then present them to the state predictor

After some initial experiments, it became clear that I needed to approach the problems separately, as I lacked a clear enough understanding of the distribution of the states to build an architecture that could accomplish both goals easily.

That left two high-level choices for the architecture (apart from the usual empircal experimentation with structure/hyperparameters/etc).

* a classic autoencoder (variational or otherwise)
* a model whose output was a simplified/compressed state (chr tables, palettes, etc) from which the image could be systematically reconstructed

I decided to try both, beginning with a classic autoencoder, experimenting to find the ideal architecture and the minimum latent dimensions necessary to get acceptable results, then comparing it to a comparable training regimen on the other architecture.

