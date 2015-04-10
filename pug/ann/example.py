"""Example pybrain network training to predict the weather

Installation:

    pip install pug-ann


Examples:
    >>> trainer = train_weather_predictor('San Francisco, CA', epochs=2, inputs=['Max TemperatureF'], outputs=['Max TemperatureF'], years=range(2013,2015), delays=(1,), use_cache=True, verbosity=0)
    >>> all(trainer.module.activate(trainer.ds['input'][0]) == trainer.module.activate(trainer.ds['input'][1]))
    False
    >>> trainer.trainEpochs(5)

    Make sure NN hasn't saturated (as it might for a sigmoid hidden layer)
    >>> all(trainer.module.activate(trainer.ds['input'][0]) == trainer.module.activate(trainer.ds['input'][1]))
    False
"""

import datetime
from pug.ann.data import weather
from pug.ann import util
from pug.nlp.util import make_date, update_dict


def train_weather_predictor(
        location='Camas, WA',
        years=range(2013, 2016,),
        delays=(1, 2, 3),
        inputs=('Min TemperatureF', 'Max TemperatureF', 'Min Sea Level PressureIn', u'Max Sea Level PressureIn', 'WindDirDegrees',),
        outputs=(u'Max TemperatureF',),
        N_hidden=6,
        epochs=30,
        use_cache=False,
        verbosity=2,
        ):
    """Train a neural nerual net to predict the weather for tomorrow based on past weather.

    Builds a linear single hidden layer neural net (multi-dimensional nonlinear regression).
    The dataset is a basic SupervisedDataSet rather than a SequentialDataSet, so the training set
    and the test set are sampled randomly. This means that historical data for one sample (the delayed
    input vector) will likely be used as the target for other samples.

    Uses CSVs scraped from wunderground (without an api key) to get daily weather for the years indicated.

    Arguments:
      location (str): City and state in standard US postal service format: "City, ST"
          alternatively an airport code like "PDX or LAX"
      delays (list of int): sample delays to use for the input tapped delay line.
          Positive and negative values are treated the same as sample counts into the past.
          default: [1, 2, 3], in z-transform notation: z^-1 + z^-2 + z^-3
      years (int or list of int): list of 4-digit years to download weather from wunderground
      inputs (list of int or list of str): column indices or labels for the inputs
      outputs (list of int or list of str): column indices or labels for the outputs

    Returns:
      3-tuple: tuple(dataset, list of means, list of stds)
          means and stds allow normalization of new inputs and denormalization of the outputs

    """
    df = weather.daily(location, years=years, use_cache=use_cache, verbosity=verbosity).sort()
    ds = util.dataset_from_dataframe(df, normalize=False, delays=delays, inputs=inputs, outputs=outputs, include_last=False, verbosity=verbosity)
    nn = util.ann_from_ds(ds, N_hidden=N_hidden, verbosity=verbosity)
    trainer = util.build_trainer(nn, ds=ds, verbosity=verbosity)
    results = trainer.trainEpochs(epochs)
    return trainer


def oneday_weather_forecast(
        location='Portland, OR',
        inputs=('Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 'Max Sea Level PressureIn', 'Mean Sea Level PressureIn', 'Min Sea Level PressureIn', 'WindDirDegrees'),
        outputs=('Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF', 'Max Humidity'),
        date=None,
        epochs=200,
        delays=(1, 2, 3, 4),
        num_years=4,
        use_cache=False,
        verbosity=1,
        ):
    """ Provide a weather forecast for tomorrow based on historical weather at that location """
    date = make_date(date or datetime.datetime.now().date())
    num_years = int(num_years or 10)
    years = range(date.year - num_years, date.year + 1)
    df = weather.daily(location, years=years, use_cache=use_cache, verbosity=verbosity).sort()
    # because up-to-date weather history was cached above, can use that cache, regardless of use_cache kwarg
    trainer = train_weather_predictor(
        location,
        years=years,
        delays=delays,
        inputs=inputs,
        outputs=outputs,
        epochs=epochs,
        verbosity=verbosity,
        use_cache=True,
        )
    nn = trainer.module
    forecast = {'trainer': trainer}

    yesterday = dict(zip(outputs, nn.activate(trainer.ds['input'][-2])))
    forecast['yesterday'] = update_dict(yesterday, {'date': df.index[-2].date()})

    today = dict(zip(outputs, nn.activate(trainer.ds['input'][-1])))
    forecast['today'] = update_dict(today, {'date': df.index[-1].date()})

    ds = util.input_dataset_from_dataframe(df[-max(delays):], delays=delays, inputs=inputs, normalize=False, verbosity=0)
    tomorrow = dict(zip(outputs, nn.activate(ds['input'][-1])))
    forecast['tomorrow'] = update_dict(tomorrow, {'date': (df.index[-1] + datetime.timedelta(1)).date()})

    return forecast


def thermostat(
        location='Camas, WA',
        days=100,
        capacity=1000,
        max_eval=1000,
        ):
    """ Control the thermostat on an AirCon system with finite thermal energy capacity (chiller)

    Useful for controlling a chiller (something that can cool down overnight and heat up during the
    hottest part of the day (in order to cool the building).

    Builds a linear single-layer neural net (multi-dimensional regression).
    The dataset is a basic SupervisedDataSet rather than a SequentialDataSet, so there may be
    "accuracy left on the table" or even "cheating" during training, because training and test
    set are selected randomly so historical data for one sample is used as target (furture data)
    for other samples.

    Uses CSVs scraped from wunderground (no api key required) to get daily weather for the years indicated.

    Arguments:
      location (str): City and state in standard US postal service format: "City, ST" or an airport code like "PDX"
      days (int): Number of days of weather data to download from wunderground
      delays (list of int): sample delays to use for the input tapped delay line.
        Positive and negative values are treated the same as sample counts into the past.
        default: [1, 2, 3], in z-transform notation: z^-1 + z^-2 + z^-3
      years (int or list of int): list of 4-digit years to download weather from wunderground
      inputs (list of int or list of str): column indices or labels for the inputs
      outputs (list of int or list of str): column indices or labels for the outputs

    Returns:
      3-tuple: tuple(dataset, list of means, list of stds)
          means and stds allow normalization of new inputs and denormalization of the outputs

    """
    pass


def explore_maze():
    # simplified version of the reinforcement learning tutorial example
    structure = [
        list('!!!!!!!!!!'),
        list('! !  ! ! !'),
        list('! !! ! ! !'),
        list('!    !   !'),
        list('! !!!!!! !'),
        list('! ! !    !'),
        list('! ! !!!! !'),
        list('!        !'),
        list('! !!!!!  !'),
        list('!   !    !'),
        list('!!!!!!!!!!'),
        ]
    structure = np.array([[ord(c)-ord(' ') for c in row] for row in structure])
    shape = np.array(structure.shape)
    environment = Maze(structure,  tuple(shape - 2))
    controller = ActionValueTable(shape.prod(), 4)
    controller.initialize(1.)
    learner = Q()
    agent = LearningAgent(controller, learner)
    task = MDPMazeTask(environment)
    experiment = Experiment(task, agent)

    for i in range(30):
        experiment.doInteractions(30)
        agent.learn()
        agent.reset()

    controller.params.reshape(shape.prod(), 4).max(1).reshape(*shape)
    # (0, 0) is upper left and (0, N) is upper right, so flip matrix upside down to match NESW action order
    greedy_policy = np.argmax(controller.params.reshape(shape.prod(), 4), 1)
    greedy_policy = np.flipud(np.array(list('NESW'))[greedy_policy].reshape(shape))
    maze = np.flipud(np.array(list(' #'))[structure])
    print('Maze map:')
    print('\n'.join(''.join(row) for row in maze))
    print('Greedy policy:')
    print('\n'.join(''.join(row) for row in greedy_policy))
    assert '\n'.join(''.join(row) for row in greedy_policy) == 'NNNNN\nNSNNN\nNSNNN\nNEENN\nNNNNN'


#################################################################
## An online (reinforcement) learning example based on the
## cart pole-balancing example in pybrian
## WIP to perform optimal control of Building HVAC system
## with limited electrical or thermal energy resource that is recharged every day

from pybrain.rl.environments import EpisodicTask
from pybrain.rl.environments.cartpole import CartPoleEnvironment
from pybrain.rl.environments.cartpole.nonmarkovpole import NonMarkovPoleEnvironment


class BalanceTask(EpisodicTask):
    """ The task of balancing some pole(s) on a cart """
    def __init__(self, env=None, maxsteps=1000, desiredValue=0, location='Portland, OR'):
        """
        :key env: (optional) an instance of a CartPoleEnvironment (or a subclass thereof)
        :key maxsteps: maximal number of steps (default: 1000)
        """
        self.location = location
        self.airport_code = weather.airport(location)
        self.desiredValue = desiredValue
        if env is None:
            env = CartPoleEnvironment()
        EpisodicTask.__init__(self, env)
        self.N = maxsteps
        self.t = 0

        # scale position and angle, don't scale velocities (unknown maximum)
        self.sensor_limits = [(-3, 3)]
        for i in range(1, self.outdim):
            if isinstance(self.env, NonMarkovPoleEnvironment) and i % 2 == 0:
                self.sensor_limits.append(None)
            else:
                self.sensor_limits.append((-np.pi, np.pi))

        # self.sensor_limits = [None] * 4
        # actor between -10 and 10 Newton
        self.actor_limits = [(-50, 50)]

    def reset(self):
        EpisodicTask.reset(self)
        self.day = weather.get_day(date='random')
        self.t = 0

    def performAction(self, action):
        self.t += 1
        EpisodicTask.performAction(self, action)

    def isFinished(self):
        if max(list(map(abs, self.env.getPoleAngles()))) > 0.7:
            # pole has fallen
            return True
        elif abs(self.env.getCartPosition()) > 2.4:
            # cart is out of it's border conditions
            return True
        elif self.t >= self.N:
            # maximal timesteps
            return True
        return False

    def getReward(self):
        angles = list(map(abs, self.env.getPoleAngles()))
        s = abs(self.env.getCartPosition())
        reward = 0
        if min(angles) < 0.05 and abs(s) < 0.05:
            reward = 0
        elif max(angles) > 0.7 or abs(s) > 2.4:
            reward = -2 * (self.N - self.t)
        else:
            reward = -1
        return reward

    def setMaxLength(self, n):
        self.N = n


from pybrain.tools.shortcuts import buildNetwork, NetworkError
from pybrain.optimization.hillclimber import HillClimber
import time
import numpy as np


def run_competition(builders=[], task=BalanceTask(), Optimizer=HillClimber, rounds=3, max_eval=20, N_hidden=3, verbosity=0):
    """ pybrain buildNetwork builds a subtly different network than build_ann... so compete them!

    buildNetwork connects the bias to the output
    build_ann does not

    build_ann allows heterogeneous layer types but the output layer is always linear
    buildNetwork allows specification of the output layer type
    """
    results = []
    builders = list(builders) + [buildNetwork, util.build_ann]

    for r in range(rounds):
        heat = []

        # FIXME: shuffle the order of the builders to keep things fair
        #        (like switching sides of the tennis court)
        for builder in builders:
            try:
                competitor = builder(task.outdim, N_hidden, task.indim, verbosity=verbosity)
            except NetworkError:
                competitor = builder(task.outdim, N_hidden, task.indim)

            # TODO: verify that a full reset is actually happening
            task.reset()
            optimizer = Optimizer(task, competitor, maxEvaluations=max_eval)
            t0 = time.time()
            nn, nn_best = optimizer.learn()
            t1 = time.time()
            heat += [(nn_best, t1-t0, nn)]
        results += [tuple(heat)]
        if verbosity >= 0:
            print([competitor_scores[:2] for competitor_scores in heat])

    # # alternatively:
    # agent = ( pybrain.rl.agents.OptimizationAgent(net, HillClimber())
    #             or
    #           pybrain.rl.agents.LearningAgent(net, pybrain.rl.learners.ENAC()) )
    # exp = pybrain.rl.experiments.EpisodicExperiment(task, agent).doEpisodes(100)

    means = [[np.array([r[i][j] for r in results]).mean() for i in range(len(results[0]))] for j in range(2)]
    if verbosity > -1:
        print('Mean Performance:')
        print(means)
        perfi, speedi = np.argmax(means[0]), np.argmin(means[1])
        print('And the winner for performance is ... Algorithm #{} (0-offset array index [{}])'.format(perfi+1, perfi))
        print('And the winner for speed is ...       Algorithm #{} (0-offset array index [{}])'.format(speedi+1, speedi))

    return results, means

try:
    # this will fail on latest master branch of pybrain as well as latest pypi release of pybrain
    from pybrain.rl.environments.mazes import Maze, MDPMazeTask
    from pybrain.rl.learners.valuebased import ActionValueTable
    from pybrain.rl.agents import LearningAgent
    from pybrain.rl.learners import Q  # , SARSA # (State-Action-Reward-State-Action)
    from pybrain.rl.experiments import Experiment
    # from pybrain.rl.environments import Task
    import pylab

    def maze():
        # import sys, time
        pylab.gray()
        pylab.ion()
        # The goal appears to be in the upper right
        structure = [
            '!!!!!!!!!!',
            '! !  ! ! !',
            '! !! ! ! !',
            '!    !   !',
            '! !!!!!! !',
            '! ! !    !',
            '! ! !!!! !',
            '!        !',
            '! !!!!!  !',
            '!   !    !',
            '!!!!!!!!!!',
            ]
        structure = np.array([[ord(c)-ord(' ') for c in row] for row in structure])
        shape = np.array(structure.shape)
        environment = Maze(structure, tuple(shape - 2))
        controller = ActionValueTable(shape.prod(), 4)
        controller.initialize(1.)
        learner = Q()
        agent = LearningAgent(controller, learner)
        task = MDPMazeTask(environment)
        experiment = Experiment(task, agent)

        for i in range(100):
            experiment.doInteractions(100)
            agent.learn()
            agent.reset()
            # 4 actions, 81 locations/states (9x9 grid)
            # max(1) gives/plots the biggest objective function value for that square
            pylab.pcolor(controller.params.reshape(81, 4).max(1).reshape(9, 9))
            pylab.draw()

        # (0, 0) is upper left and (0, N) is upper right, so flip matrix upside down to match NESW action order
        greedy_policy = np.argmax(controller.params.reshape(shape.prod(), 4), 1)
        greedy_policy = np.flipud(np.array(list('NESW'))[greedy_policy].reshape(shape))
        maze = np.flipud(np.array(list(' #'))[structure])
        print('Maze map:')
        print('\n'.join(''.join(row) for row in maze))
        print('Greedy policy:')
        print('\n'.join(''.join(row) for row in greedy_policy))

        # pylab.show()
except ImportError:
    pass


if __name__ == '__main__':
    try:
        explore_maze()
    except:
        
    import sys
    print(run_competition(verbosity=0))
    sys.exit(0)
if __name__ == "__main__":
    try:
    except:
        from traceback import format_exc
        sys.exit(format_exc())
