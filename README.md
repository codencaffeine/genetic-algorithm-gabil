### Machine Learning HW4 - Genetic Algorithm 

Package requirememt: `numpy` 

#### Step to run the program:  
1. There are **four** test scripts in `root` dir for part (d) - i,ii, iii, iv
   *  `python3 testTennis.py`
   *  `python3 testIris.py`
   *  `python3 testIrisSelection.py`
   *  `python3 testIrisReplacement.py`

#### Information
1. Main Genetic Algorithm implementation script is `genetic_algorithm.py`
2. To enable plotting (needs matplotlib), add the argument `plotOn=True` to Genetic class object initialization
3. The class init function contains docstring to explain all the possible parameters and their defaults
4. Logs for part d.i and d.ii, are present in `./logs` dir
5. Plots naming convention: `<population_size>_<max_rules_allowed>_<generation_no>_<data_name>.png`  
   * `500_5_150_iris.png`
6. `iris_rules.png` and `tennis_rules.png` shows train and test accuracy along with Human-readable learned rules