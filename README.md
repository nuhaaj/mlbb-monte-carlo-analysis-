this project uses **monte carlo simulation** to look at how much randomness can affect ranked games in mobile legends (mlbb).

instead of judging performance from just a few matches, the code simulates 10,000 possible match outcomes based on a fixed true win rate.

## assumptions
to keep the model simple:

* each match is independent
* true win probability stays constant during each simulation
* each win gives **+1 star**
* each loss gives **-1 star**
* the target is **+50 net stars**
* star protection and bonus stars are excluded
* matchmaking difficulty is assumed to stay the same

## simulation setup
baseline simulation:

* true win probability: **55%**
* matches per season: **200**
* number of simulations: **10,000**
* rank target: **+50 stars**

i also tested:

* win probabilities from **45% to 65%**
* match horizons of **100, 200, 300 and 500 matches**

## methodology
each match is randomly simulated as either a win or loss based on the player's true win probability.

the process is repeated across **10,000 simulated seasons** and the model records:

* observed win rate
* longest win and loss streak
* probability of **5+, 7+ and 10+ loss streaks**
* net star progression
* probability of reaching **+50 stars**
* number of matches needed to reach the target
* maximum rank drawdown

the model also compares simulated results with theoretical values to check whether the simulation behaves as expected.

## analysis
the project focuses on four main areas:

### 1. win rate uncertainty

how much can a player's observed win rate differ from their true win probability just because of random match outcomes?

### 2. losing streak risk

how common are long losing streaks even for players with a positive true win rate?

### 3. rank climbing

how likely is a player to reach **+50 stars**, and how many matches might it take?

### 4. drawdown risk

how far can a player fall from their previous highest star position before recovering?

## sensitivity analysis

i also changed the model inputs to see how the results change across different:

* **true win probabilities**
* **numbers of matches**

this helps show how player strength and time spent playing affect the probability of reaching the rank target.

## visualisations
the project generates charts for:

* losing streak probability vs true win rate
* observed win rate distribution
* probability of reaching +50 stars
* simulated vs theoretical rank progression
* rank drawdown risk
* match horizon sensitivity

## results
the main simulation results and charts are available in the output folder.

the analysis includes:

* baseline simulation results
* win rate scenarios
* rank scenarios
* time horizon sensitivity
* full simulation datasets

## limitations
this is a simplified model and is not meant to perfectly recreate mlbb matchmaking.

the model does not include things such as:

* changes in matchmaking difficulty
* hero selection and team composition
* differences between teammates and opponents
* star protection or bonus stars
* player tilt or changes in performance over time

because of this, the project mainly looks at **randomness under controlled assumptions**, rather than predicting exactly what will happen to a real player.

## tools used
* python
* numpy
* pandas
* matplotlib
* excel
