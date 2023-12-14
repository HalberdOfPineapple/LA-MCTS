# LA-MCTS Experiment Notes

## TODO

- Further make clear the running logic of this repository
  - with a focus on how to implement a new sampler and introduce new benchmark task
  - and read SMAC3 code for ways to introduce SMAC as a new sampler
- find the reason why MuJoCo cannot be successfully launched for running
- optional potential optimization: enabling better parameter passing mechanism through yaml/json/ini



### Reproduction of Paper's Experiments

#### Setup RL environment

#### Make clear original settings



#### How to use no-MCTS approaches under this framework?

- Easy to write a procedure without MCTS involved, where only the sample bag is kept being updated and sampler does work on the whole sample bag
- Or we can directly use external tools (e.g. SMAC3) to do experiments using conventional BO-methods for the same benchmark



### Integrating SMAC as a new sampler

The reference is `bo_sampler` and the logic is not that difficult

The general workflow can be summarized as:

- build the basic `sample_bag` to contain the samples
- consider the input bag as `gp_bag` for fitting GP
- enter the loop for sampling
  - Pick a number of samples from `gp_bag` for training GP
  - Build scaler, kernel function and GPR model
  - Fit the model to scaled data
  - Random sampling a number of candidates (`_gp_num_cands`) after filing out unmatched ones by `path.filter`
  - Use the trained GP model for estimating the performance for the sampled candidates
  - Select the batch of samples (`batch_size`) maximizing acquisition function 
  - Evaluate the batch of samples using the function
  - Add the newly evaluated samples (`batch_size`) to `gp_bag` and `sample_bag`
    - `gp_bag` will be used in the next iteration for fitting GP
- The loop will terminate until we sampled `num_samples` (input) samples and `sample_bag` will be returned

**TODO: Now to look at the procedure of using SMAC3** 



## Reason for choosing this version

- this version of code is more comprehensive and better written (with respect to the functionality modularization)
- TurBO is missing in the original version of code and the sampling and classification logic is coupled



## MuJoCo Environment Settings

### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

### Python Install

```sh
pip install -U 'mujoco-py<2.2,>=2.1'
```

### Problems

#### Cpython

https://github.com/openai/mujoco-py/issues/773

```
pip install "cython<3"
```



#### `GL/osmesa.h`

```sh
sudo apt-get install libosmesa6-dev
```



#### `patchelf`

https://github.com/openai/mujoco-py/issues/652

```
sudo apt-get install patchelf
```



#### `GLIBCXX_3.4.30` not found

https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin

```
conda install -c conda-forge gcc=12.1.0
```

which updates the system's `gcc` to version 12.1.0





## Modification Logs

### Change `gen_sample_bag` calling

The original code uses `self._func.gen_sample_bag` which will not update the statistical wrapper's data and will not exit the search loop.

For this problem, I wrapped and added `gen_sample_bag` to `StatsFuncWrapper` like below:

```python
def gen_sample_bag(self, xs: Optional[np.ndarray] = None) -> Bag:
   if xs is None or len(xs) == 0:
       return Bag(self._func.dims,
                  is_minimizing=self._func.is_minimizing,
                  is_discrete=self._func.is_discrete)
        fs, features = self(self._func.transform_input(xs))
   return Bag(xs, fs, features, self._func.is_minimizing, self._func.is_discrete)
```

where each time of function calling will be wrapped by the `StatsFuncWrapper` (`__call__` is made through the wrapper).

And I change `self._func.gen_sample_bag` to `self._func_stat.gen_sample_bag` in all sampler classes.







## Workflow

using `example_levy.py`

### General

#### Function and Parameters Definition

```python
func = Levy()
func_wrapper = StatsFuncWrapper(func)
params = func.mcts_params(SamplerEnum.TURBO_SAMPLER, ClassifierEnum.THRESHOLD_SVM_CLASSIFIER)
```

which includes:

- `__call__` includes function execution logic, returning the evaluation result
- `mcts_params`: taking a sampler and classifier as inputs, **modify the parameters of MCTS, sampler and classifier** given the function context

`func_wrapper` is the decorator for doing statistics on function execution



#### MCTS Creation

```
mcts = MCTS.create_mcts(func, func_wrapper, params)
```



#### MCTS Search

```python
st = time.time()
try:
	mcts.search(greedy=GreedyType.ConfidencyBound, call_budget=call_budget)
except TimeoutError:
	raise TimeoutError("Search failed: Timeout")
wt = time.time() - st
```



### MCTS Initialization

#### Sampler Initialization

<img src="C:\Users\Viktor\AppData\Roaming\Typora\typora-user-images\image-20231214141907952.png" alt="image-20231214141907952" style="zoom:33%;" />

The sampler needs to take the function, function's statistics wrapper and sampler-related parameters as inputs

```
"sampler": {
        "type": sampler,
        "params": {
        	  "acquisition": "ei",
        	  "nu": 2.5,
        	  **SAMPLER_PARAMS[sampler]
       	}
	}
```

The parameters for each specific sampler is included in `SAMPLER_PARAMS` of `config.py`

```python
SAMPLER_PARAMS = {
    SamplerEnum.RANDOM_SAMPLER: {
    },
    SamplerEnum.BO_SAMPLER: {
        "acquisition": "ei",
        "nu": 1.5,
        "gp_num_cands": 10000,
        "gp_max_samples": 0,
        "batch_size": 0
    },
    SamplerEnum.TURBO_SAMPLER: {
        "acquisition": "ei",
        "nu": 1.5,
        "gp_max_samples": 0,
        "gp_num_cands": 5000,
        "batch_size": 5,
        "fail_threshold": 10,
        "succ_threshold": 3,
        "device": "cuda"
    },
    SamplerEnum.NEVERGRAD_SAMPLER: {
        "sample_center": SampleCenter.Mean
    },
    SamplerEnum.CMAES_SAMPLER: {
    },
}
```





#### Classifier Initialization

<img src="C:\Users\Viktor\AppData\Roaming\Typora\typora-user-images\image-20231214142227971.png" alt="image-20231214142227971" style="zoom:33%;" />

similar to sampler's initialization

But note that for a classifier, what needs to be passed is a factory for creating a specific type of classifier at each node



#### MCTS Object Initialization

```python
MCTS(
	func=func, 
    func_stats=func_stats,
    **params["params"], 
    sampler=sampler,
    classifier_factory=classifier_factory
)
```

where MCTS-specific parameters include:

```python
MCTS_PARAMS = {
    "cp": 0.1, # parameter controlling exploration
    "cb_base": ConfidencyBase.Best, # confident bound base, 0: mean, 1: best
    "leaf_size": 10, # number of samples hold by a leaf
    "num_init_samples": 100,
    "num_samples_per_sampler": 20,
    "search_type": SearchType.Vertical,
    "num_split_worker": 1
}
```

(which can be modified for the function when doing `func.mcts_params`)

- Setup the parameters including
  - `_func`, and `_func_stats`
  - `_num_init_samples` - number of samples for initialization
  - `_cp` - controlling exploration
  - `cb_base` confidence based on best or mean (**?**)
  - `leaf_size` the threshold of number of samples for a leaf or internal node
  - `_num_samples_per_sampler`: number of samples being obtained each time
  - `_num_split_worker` - number of threads for building the tree



### MCTS Search

```python
st = time.time()
try:
	mcts.search(greedy=GreedyType.ConfidencyBound, call_budget=call_budget)
except TimeoutError:
	raise TimeoutError("Search failed: Timeout")
wt = time.time() - st
```

#### Node class initialization

```python
Node.init(self._num_split_worker)
```

which basically set up the parameters in node class:

<img src="C:\Users\Viktor\AppData\Roaming\Typora\typora-user-images\image-20231214143313751.png" alt="image-20231214143313751" style="zoom:33%;" />

especially for multiple programming when `_num_split_workers` is greater than 1

#### Record Start Time

```python
start_time = time.time()
```



#### Tree Building

```
self.init_tree()
self._mcts_stats = self._stats()
```

including:

- sampling `self._num_init_samples` samples randomly

  - ```python
    self._sampler.sample(self._num_init_samples)
    ```

- Build root node with function dimensions, leaf size

  - ```python
     self._root = Node(
    	self._func.dims,  self._leaf_size, self._cp * samples.best.fx, 
        self._classifier_maker, # classifier is also encapsulated in a Node
        samples, self._cb_base)
    ```

- Build the tree from the root:

  - see static method `Node.build_tree`, which encapsulate the **full logic of building the LA-MCTS**



#### Optimization Iterations (Vertical Search case)

At each iteration:

- the leaves will be sorted by (c.confidence_bound, c.mean, c.best.fx)

  - ```python
    all_leaves = []
    self._root.sorted_leaves(all_leaves, greedy)
    ```

  - this function will put the sorted leaves into the input list (`all_leaves`)

- Sample from the leftmost leaf nodes to the rightmost (**with an extra path argument for `sample` function**)

  - <img src="C:\Users\Viktor\AppData\Roaming\Typora\typora-user-images\image-20231214152103597.png" alt="image-20231214152103597" style="zoom:33%;" />
  - the results of sampling will be stored in `new_samples`. The sampling will be terminated when new samples reach the limit for each sampling

- Add the new samples to the root (the whole search space)

  - ```
    self._root.add_bag(new_samples)
    ```

- 









### Structure: Node

#### Initialization

- 



### Structure: Bag

Bag essentially stands for **search region/subspace, represented by a bag of sample data falling into it**, which includes the following basic members:

- `_xs: Union[np.ndarray, int]` - array of sample data (configurations)
- `_fxs: np.ndarray` - array of evaluation results corresponding to `_xs`  
  - there is an assert `fxs.shape == (self._xs.shape[0],)`
- 
