# blipcon

**Generative AI producing novel interactive classic-style games.**

This is a learning project aimed at exploring deep learning techniques for predicting video game states using transformer-like models. The original objective was to create a model that can accurately predict game states based on current state and action inputs. However, as the goal was learning I let myself go down some fairly deep rabbit holes.

## Architectures Considered

There are three parts to the problem

* encode game states
* predict next state given previous state-action pair(s)
* embed the learning in a way that can be leveraged to generate novel sequences (ie, original games)

The architecture depends on the approach to the third, which is the least well-understood. I considered three approaches

* present a RL model with a VM & headless DSL; train it to reproduce observed states procedurally by manipulating an AST (ie, according to the rules of a "game" of game-making); outcome here is a playable game (versus a game generated on the fly)

* train the model as a hypernetwork to produce the weights for a "state machine" model, whose scope is limited to mapping (s1, a) -> s2. Hypothetically, the weights (or specifically, the latents from which the weights would be decoded) would represent the "code" or "rules" of the state machine. outcome here still has a model in the loop, but a theoretically maximally efficient one

* build a transformer to predict states from sequences of state-action pairs; outcome here is the least efficient, but likely the most successful (this approach seemed the most well-understood)

Given that all approaches would require a working state predictor to succeed, and some initial tests wih the hypernetwork [here](early_tests/complex.py) and [here](early_tests/simple.py) that demonstrated that it had more hyperparameters than I had resources to tune, it made sense to attempt to focus on building a succesful state predictor first.

That problem presented two approaches to embedding states

* train the transformer to encode/decode the states it produces
* train a separate model to encode the states, then present them to the state predictor

After some initial experiments, it became clear that I needed to approach the problems separately, as I lacked a clear enough understanding of the distribution to guess at an ideal encoding architecture.

That left two high-level choices for the ae architecture

* a classic autoencoder (variational or otherwise)
* a model whose output was a simplified/compressed state (chr tables, palettes, etc) from which the image could be systematically reconstructed

I decided to try both, beginning with a classic autoencoder, experimenting to find the ideal architecture and the minimum latent dimensions necessary to get acceptable results, then comparing it to a comparable training regimen on the other architecture. Likewise, I decided to first build a model capable of predicting actions alone, then extend it to predict states. (The latter decision was maybe not ideal, as human play produced a wider variety of states.)

### Components

This code is a mess, but roughly there is

* a lua script for streaming frames from a game emulator to a pipe or service
* utilities for turing the raw states into usable training data
* a /predict service for predicting best next *actions* from a game state (predicting next states were going to come later)
* various autoencoder component modules; roughly a generic AE, VAE, and a specialized "8-bit graphics AE" (see below),
* two approaches to the RL model; a calssic double-Q learner and a transformer-based model

### Encoding the states

For both of the AEs, I trained them with a variety of images randomly sampled from human play across about 8 top-ranked NES games.

#### Classic AE approach

A simple AE without reparametization worked better than a VAE. Single-headed self-attention (modeled after stable diffusion's autoencoder) greatly improved performance. Fewer layers and aggressive downsampling actually seemed to perform better. But ultimately this approach is *lossy*, and pixel art does not like that. My theory is that the pixels in pixel art, versus conventional art, form a discrete distribution, or at least have enough in common with one that they have to be treated differently. (An icon is chosen partly for its representational qualities, partly for its distinctness from other icons, etc.)

#### Compositional AE approach

The first attempt at a [structure-aware autoencoder](modules/nes.py) failed miserably. The idea was to separately learn

* the underlying 8x8 four-color patterns
* the palettes used to colorize them
* the tables governing the pattern layout
* the tables governing the palette application

then to compose them with differentiable functions to produce the image. This required multiple passes through a gumble-softmax transformation to extract the indexes for the pattern and palette tables. The results were mostly noise forming a rough approximation of overall brightness of the image. My theory is that this was simply too many moving parts for the model to learn -- especially expecting the model to learn the correct composition of multiple random samples.

It's possible that this could be successful if approached in supervised stages, which might be worth revisting if could lead to generating novel images that compose into crisp, stylistically consistent images. (However, it is also likely that I could do just as well by tuning the classic AE to produce crisper images.)

### Classic versus Transformer RL

I did the initial training first on an NES emulator running Mario Brothers. This is well-trod territory with openai gym, but A) most examples I could find were broken and B) they gamed things by simplifying the state and action space. I wanted to see if I could get a model to learn to play the game from raw pixels, since my ultimate goal was to create novel playable games.

The classic approach got stuck when anything resembling strategic behavior was required. (Like running to jump over an especially tall pipe.) This was obviously due to the action space being too open ended (random button mashing) for the model to accidentally discover useful behavior. It's possible simplifying to something more like "start jumping, stop running, etc" would improve the likelihood.

The transformer was a mixed bag. It also got stuck initially, unless primed with some human play, but it tended to get stuck in repetitive fruitless action, like trying to jump backwards against the left side of the screen, or gaming its reward function by inching the screen forward when otherwise stuck.

I stopped before taking the next step of abandoning the RL approach and focusing on training the transformer to predict states. (Mainly because I have a job.)

### Next Steps

I'm curious about experimenting with a state predictor that does hybrid state prediction-reinforcement learning batched in episodes. That is, "input" would always consist of the events preceding the beginning of an "episode" that leads to a significant reward. The expected "output" would be the states composing the episode itself. (Ie, the situation is the prompt, the entire strategy, rather than a single action, is the response.) This would have to be exclusively trained on human play, at least initially.

I have no idea if this is even remotely novel or likely to work. But I'd be curious to see if it at least outperformed the original approach.